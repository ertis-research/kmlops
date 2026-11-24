"""Input/output features that all the versions of a model must share.

Every training registers a version of the model `model_name` in the MLflow Model Registry
(experiments and models are independent: several experiments can feed one model). A new training
must have the same input and output features (name and type) as the latest version of its model.
Otherwise a consumer of "the latest version" could get a model that expects other inputs or
predicts something else.

The features of the existing versions are taken from two places, and the new training must
match both:
- the latest completed or in-progress training of the model in the kmlops database, in any
  experiment (failed and cancelled trainings never produced a version);
- the latest version of the registered model in MLflow, which also covers models with no rows
  in the kmlops database. Its features are read from the run parameters that Ludwig logs
  (`input_features` / `output_features`) and, if they are missing, from the
  `model_hyperparameters.json` of the logged model.

Feature order and per-feature settings (encoders, preprocessing...) can change between versions.
The database allows one training in progress per model, so two trainings with different features
cannot pass this check at the same time.
"""
import ast
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

from src.schemas import ACTIVE_STATUSES

REFERENCE_STATUSES = ("completed", *ACTIVE_STATUSES)

FeatureSignature = Dict[str, Dict[str, str]]  # {"input_features": {name: type}, "output_features": {...}}
FEATURE_SECTIONS = ("input_features", "output_features")

# Old Ludwig type names, as found in models trained with previous versions.
TYPE_ALIASES = {"numerical": "number"}

# Relative to the source of a model version logged by Ludwig.
HYPERPARAMETERS_PATH = "model/model_hyperparameters.json"


class InvalidFeatures(ValueError):
    """The Ludwig config does not define valid input_features / output_features."""


class FeatureCheckUnavailable(Exception):
    """The existing versions could not be read from MLflow, so the check cannot be done."""


class IncompatibleFeatures(Exception):
    """The features differ from those of the existing versions of the model."""

    def __init__(self, model_name: str, reference: str, differences: List[str]) -> None:
        self.model_name = model_name
        self.reference = reference
        self.differences = differences
        super().__init__(
            f"The input and output features must match those of the existing versions of model '{model_name}' "
            f"({reference}): {'; '.join(differences)}. "
            f"Use another model_name to train a model with different features."
        )


def feature_signature(config: Any) -> FeatureSignature:
    """{section: {feature name: type}} of a Ludwig config. Raises InvalidFeatures."""
    if not isinstance(config, dict):
        raise InvalidFeatures("config_json must be a JSON object")
    signature: FeatureSignature = {}
    for section in FEATURE_SECTIONS:
        features = config.get(section)
        if not isinstance(features, list) or not features:
            raise InvalidFeatures(f"config_json must define '{section}' as a non-empty list")
        by_name: Dict[str, str] = {}
        for feature in features:
            name = feature.get("name") if isinstance(feature, dict) else None
            ftype = feature.get("type") if isinstance(feature, dict) else None
            if not isinstance(name, str) or not name.strip() or not isinstance(ftype, str) or not ftype.strip():
                raise InvalidFeatures(f"Every entry of '{section}' must have a 'name' and a 'type'")
            if name in by_name:
                raise InvalidFeatures(f"Duplicated feature '{name}' in '{section}'")
            ftype = ftype.strip().lower()
            by_name[name] = TYPE_ALIASES.get(ftype, ftype)
        signature[section] = by_name
    return signature


def describe_differences(expected: FeatureSignature, got: FeatureSignature) -> List[str]:
    differences = []
    for section in FEATURE_SECTIONS:
        old, new = expected[section], got[section]
        missing = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        changed = sorted(n for n in set(old) & set(new) if old[n] != new[n])
        if missing:
            differences.append(f"{section} missing {missing}")
        if added:
            differences.append(f"{section} not in the existing versions {added}")
        for name in changed:
            differences.append(f"{section} '{name}' has type '{new[name]}' instead of '{old[name]}'")
    return differences


def _signature_or_none(config: Any, where: str) -> Optional[FeatureSignature]:
    try:
        return feature_signature(config)
    except InvalidFeatures as e:
        print(f"Feature check: {where} has no valid features ({e}); not used as reference.")
        return None


# ---- references ----------------------------------------------------------------


def _local_reference(store: Any, model_name: str) -> Optional[tuple]:
    training = store.latest_model_training(model_name, REFERENCE_STATUSES)
    if training is None:
        return None
    where = f"training {training['training_id']} of experiment '{training['experiment_name']}'"
    signature = _signature_or_none(training.get("config"), where)
    return (where, signature) if signature is not None else None


def _is_not_found(e: MlflowException) -> bool:
    return e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def _signature_from_run_params(client: MlflowClient, run_id: str) -> Optional[FeatureSignature]:
    """Ludwig logs input_features / output_features as the repr of the config lists."""
    try:
        params = client.get_run(run_id).data.params
    except MlflowException as e:
        if _is_not_found(e):
            return None
        raise
    if not all(section in params for section in FEATURE_SECTIONS):
        return None
    try:
        config = {section: ast.literal_eval(params[section]) for section in FEATURE_SECTIONS}
    except (ValueError, SyntaxError):
        return None  # e.g. a value truncated by MLflow's parameter length limit
    return _signature_or_none(config, f"run {run_id}")


def _signature_from_model_artifact(source: str) -> Optional[FeatureSignature]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{source.rstrip('/')}/{HYPERPARAMETERS_PATH}", dst_path=tmp
            )
            hyperparameters = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # missing artifact, model not logged by Ludwig, unsupported source...
        print(f"Feature check: could not read {HYPERPARAMETERS_PATH} from {source}: {e}")
        return None
    return _signature_or_none(hyperparameters, f"model {source}")


def _mlflow_reference(model_name: str) -> Optional[tuple]:
    """Features of the latest version of the registered model."""
    client = MlflowClient()
    try:
        model = client.get_registered_model(model_name)
        if not model.latest_versions:
            return None
        version = max(model.latest_versions, key=lambda v: int(v.version))
        signature = _signature_from_run_params(client, version.run_id) if version.run_id else None
    except MlflowException as e:
        if _is_not_found(e):
            return None
        raise FeatureCheckUnavailable(f"Could not read the versions of model '{model_name}' from MLflow: {e}") from e
    except Exception as e:
        raise FeatureCheckUnavailable(f"Could not read the versions of model '{model_name}' from MLflow: {e}") from e

    if signature is None and version.source:
        signature = _signature_from_model_artifact(version.source)
    if signature is None:
        # A model registered outside kmlops, without Ludwig params or artifacts: nothing to compare with.
        print(f"Feature check: the features of version {version.version} of model '{model_name}' are unknown; not checked.")
        return None
    return f"model version {version.version}", signature


def check_compatible_features(store: Any, model_name: str, config: Any, mlflow_on: bool) -> None:
    """Raises InvalidFeatures, IncompatibleFeatures or FeatureCheckUnavailable."""
    signature = feature_signature(config)
    references = [_local_reference(store, model_name)]
    if mlflow_on:
        references.append(_mlflow_reference(model_name))
    for reference in references:
        if reference is None:
            continue
        where, expected = reference
        differences = describe_differences(expected, signature)
        if differences:
            raise IncompatibleFeatures(model_name, where, differences)
