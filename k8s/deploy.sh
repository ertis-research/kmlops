#!/usr/bin/env bash
# Deploys kmlops to Kubernetes using the values of a .env file.
#
#   k8s/deploy.sh [ENV_FILE]          (default: .env in the repository root)
#   DRY_RUN=1 k8s/deploy.sh           (validate everything, change nothing, build nothing)
#   SKIP_BUILD=1 k8s/deploy.sh        (deploy the image already in the registry)
#
# From the .env file it:
#   0. builds the kmlops image from the repository and pushes it as KMLOPS_IMAGE
#      (with your own "docker login": REGISTRY_* is a pull-only account);
#   1. creates/updates the "kmlops-minio-credentials" Secret with kmlops's configuration;
#   2. creates/updates the "kmlops-registry-creds" pull Secret, if REGISTRY_* are set;
#   3. applies k8s/rbac.yaml and k8s/deployment.yaml (with KMLOPS_IMAGE as the image)
#      in the namespace KUBERNETES_NAMESPACE;
#   4. restarts kmlops so it picks up the new configuration and image.
#
# Secrets are written with create/replace, never "kubectl apply", so their values are not
# copied into the "last-applied-configuration" annotation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$ROOT/.env}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

CONFIG_SECRET="kmlops-minio-credentials"
PULL_SECRET="kmlops-registry-creds"
DEPLOYMENT="kmlops"

# Variables used only by this script or only for local development: never copied to the cluster.
#  - SQLITE_PATH: the local path (e.g. ./data/kmlops.db) would put the database outside the volume;
#    in the cluster kmlops uses its default, /data/kmlops.db, which is the persistent volume.
#  - KUBERNETES_CONFIG_PATH: inside the cluster kmlops uses its service account.
#  - MINIO_CREDENTIALS_SECRET and its *_KEY_KEY variables: deployment.yaml fixes the Secret to
#    $CONFIG_SECRET because this script creates it with the default keys (MINIO_ACCESS_KEY,
#    MINIO_SECRET_KEY); a local .env must not override that.
NOT_FOR_CLUSTER="KMLOPS_IMAGE REGISTRY_SERVER REGISTRY_USERNAME REGISTRY_PASSWORD SQLITE_PATH KUBERNETES_CONFIG_PATH MINIO_CREDENTIALS_SECRET MINIO_CREDENTIALS_ACCESS_KEY_KEY MINIO_CREDENTIALS_SECRET_KEY_KEY"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

[[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE (copy .env.example to .env and fill it in)"
command -v kubectl >/dev/null || die "kubectl is not installed or not in PATH"
if [[ "$DRY_RUN" != "1" && "$SKIP_BUILD" != "1" ]]; then
    command -v docker >/dev/null || die "docker is not installed or not in PATH (or use SKIP_BUILD=1 to deploy the image already in the registry)"
    docker info >/dev/null 2>&1 || die "docker is not running (start Docker Desktop, or use SKIP_BUILD=1)"
fi

# ---- Parse the .env file (same rules as python-dotenv for the usual cases) ----------
declare -A ENV=()
declare -a KEYS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                                   # Windows line endings
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue        # comments and blank lines
    line="${line#"${line%%[![:space:]]*}"}"                # leading spaces
    line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; key="${key%"${key##*[![:space:]]}"}"
    value="${line#*=}"; value="${value#"${value%%[![:space:]]*}"}"
    if [[ "$value" =~ ^\"(.*)\"[[:space:]]*(#.*)?$ ]] || [[ "$value" =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then
        value="${BASH_REMATCH[1]}"                         # quoted value
    else
        value="${value%%[[:space:]]#*}"                    # unquoted: drop inline comment
        value="${value%"${value##*[![:space:]]}"}"
    fi
    [[ -n "${ENV[$key]+x}" ]] || KEYS+=("$key")
    ENV[$key]="$value"
done < "$ENV_FILE"

get() { echo "${ENV[$1]:-}"; }

NAMESPACE="$(get KUBERNETES_NAMESPACE)"
IMAGE="$(get KMLOPS_IMAGE)"
[[ -n "$NAMESPACE" ]] || die "KUBERNETES_NAMESPACE is not set in $ENV_FILE"
[[ -n "$IMAGE" ]] || die "KMLOPS_IMAGE is not set in $ENV_FILE (the kmlops image to deploy, e.g. <registry>/<project>/kmlops:latest)"
for required in MINIO_URL MINIO_ACCESS_KEY MINIO_SECRET_KEY BUCKET_NAME; do
    [[ -n "$(get $required)" ]] || die "$required is not set in $ENV_FILE (kmlops would not start)"
done

KUBECTL=(kubectl -n "$NAMESPACE")
if [[ "$DRY_RUN" == "1" ]]; then
    info "DRY RUN: nothing will be changed in the cluster"
fi
info "Namespace: $NAMESPACE"
info "Image:     $IMAGE"
"${KUBECTL[@]}" get secrets >/dev/null 2>&1 \
    || die "namespace '$NAMESPACE' does not exist or you have no access to it (check KUBERNETES_NAMESPACE and your kubeconfig)"

# ---- 0. Build and push the image ---------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
    info "Image build: skipped (dry run)"
elif [[ "$SKIP_BUILD" == "1" ]]; then
    info "Image build: skipped (SKIP_BUILD=1), deploying the image already in the registry"
else
    info "Building $IMAGE from $ROOT"
    docker build -t "$IMAGE" "$ROOT" || die "docker build failed"
    info "Pushing $IMAGE"
    docker push "$IMAGE" \
        || die "docker push failed. Log in with an account that can push to that project: docker login ${IMAGE%%/*} (REGISTRY_* is pull-only)"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
chmod 700 "$TMP_DIR"

# Creates or replaces a Secret from a manifest read on stdin.
upsert_secret() {
    local name="$1" manifest="$TMP_DIR/$1.yaml"
    cat > "$manifest"
    if [[ "$DRY_RUN" == "1" ]]; then
        "${KUBECTL[@]}" create --dry-run=server -f "$manifest" >/dev/null 2>&1 \
            || "${KUBECTL[@]}" replace --dry-run=server -f "$manifest" >/dev/null
        echo "    $name: would be created/updated"
    elif "${KUBECTL[@]}" get secret "$name" >/dev/null 2>&1; then
        "${KUBECTL[@]}" replace -f "$manifest" >/dev/null
        echo "    $name: updated"
    else
        "${KUBECTL[@]}" create -f "$manifest" >/dev/null
        echo "    $name: created"
    fi
}

# ---- 1. Configuration Secret -----------------------------------------------------
info "Secret $CONFIG_SECRET (kmlops configuration)"
CONFIG_ENV="$TMP_DIR/config.env"
: > "$CONFIG_ENV"
for key in "${KEYS[@]}"; do
    [[ " $NOT_FOR_CLUSTER " == *" $key "* ]] && continue
    printf '%s=%s\n' "$key" "${ENV[$key]}" >> "$CONFIG_ENV"
done
echo "    keys: $(cut -d= -f1 "$CONFIG_ENV" | tr '\n' ' ')"
kubectl create secret generic "$CONFIG_SECRET" --namespace "$NAMESPACE" \
    --from-env-file="$CONFIG_ENV" --dry-run=client -o yaml | upsert_secret "$CONFIG_SECRET"

# ---- 2. Registry pull Secret ------------------------------------------------------
info "Secret $PULL_SECRET (registry credentials)"
REGISTRY_SERVER="$(get REGISTRY_SERVER)"; REGISTRY_USERNAME="$(get REGISTRY_USERNAME)"; REGISTRY_PASSWORD="$(get REGISTRY_PASSWORD)"
if [[ -n "$REGISTRY_SERVER" && -n "$REGISTRY_USERNAME" && -n "$REGISTRY_PASSWORD" ]]; then
    echo "    registry: $REGISTRY_SERVER, user: $REGISTRY_USERNAME"
    kubectl create secret docker-registry "$PULL_SECRET" --namespace "$NAMESPACE" \
        --docker-server="$REGISTRY_SERVER" --docker-username="$REGISTRY_USERNAME" \
        --docker-password="$REGISTRY_PASSWORD" --dry-run=client -o yaml | upsert_secret "$PULL_SECRET"
elif [[ -n "$REGISTRY_SERVER$REGISTRY_USERNAME$REGISTRY_PASSWORD" ]]; then
    die "set all of REGISTRY_SERVER, REGISTRY_USERNAME and REGISTRY_PASSWORD, or none of them"
elif "${KUBECTL[@]}" get secret "$PULL_SECRET" >/dev/null 2>&1; then
    echo "    REGISTRY_* not set: keeping the existing $PULL_SECRET"
else
    die "$PULL_SECRET does not exist in '$NAMESPACE' and REGISTRY_* are not set in $ENV_FILE: the image could not be pulled"
fi

# ---- 3. Manifests -----------------------------------------------------------------
info "Manifests"
DEPLOY_MANIFEST="$TMP_DIR/deployment.yaml"
# Replace the container image (the only indented "image:" line) with KMLOPS_IMAGE.
sed -E "s|^([[:space:]]+image:[[:space:]]).*|\1${IMAGE//|/\\|}|" "$ROOT/k8s/deployment.yaml" > "$DEPLOY_MANIFEST"
grep -qE "^[[:space:]]+image: ${IMAGE//./\\.}$" "$DEPLOY_MANIFEST" || die "could not set the image in deployment.yaml"

EXISTED=0
"${KUBECTL[@]}" get deployment "$DEPLOYMENT" >/dev/null 2>&1 && EXISTED=1
if [[ "$DRY_RUN" == "1" ]]; then
    "${KUBECTL[@]}" apply --dry-run=server -f "$ROOT/k8s/rbac.yaml" -f "$DEPLOY_MANIFEST"
    info "DRY RUN finished: no changes were made"
    exit 0
fi
"${KUBECTL[@]}" apply -f "$ROOT/k8s/rbac.yaml" -f "$DEPLOY_MANIFEST"

# ---- 4. Restart so kmlops reads the new Secrets (and pulls :latest again) -----------
if [[ "$EXISTED" == "1" ]]; then
    info "Restarting $DEPLOYMENT"
    "${KUBECTL[@]}" rollout restart "deployment/$DEPLOYMENT"
fi
info "Waiting for $DEPLOYMENT"
if ! "${KUBECTL[@]}" rollout status "deployment/$DEPLOYMENT" --timeout=300s; then
    echo
    "${KUBECTL[@]}" get pods -l app=kmlops
    die "kmlops did not become ready. Check: kubectl -n $NAMESPACE describe pod -l app=kmlops (see docs/runbook.md)"
fi
info "kmlops is running in '$NAMESPACE'"
