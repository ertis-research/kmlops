"""Deterministic generator for the example datasets (stdlib only).

Usage: python examples/generate_examples.py
Writes data.csv into each dataset folder next to this script. Seed: 42.
"""

import csv
import math
import os
import random

SEED = 42
ROWS = 1000
HERE = os.path.dirname(os.path.abspath(__file__))


def _write(folder, header, rows):
    path = os.path.join(HERE, folder, "data.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def binary_failure(rng):
    offsets = {"A": 0.0, "B": 0.6, "C": 1.2}
    scored = []
    for _ in range(ROWS):
        rpm = rng.gauss(1800, 300)
        temp = rng.gauss(70, 10)
        vib = abs(rng.gauss(2.0, 0.8))
        pres = rng.gauss(100, 15)
        mt = rng.choice("ABC")
        score = (
            (rpm - 1800) / 300 * 0.8
            + (temp - 70) / 10 * 0.9
            + (vib - 2.0) / 0.8 * 1.0
            - (pres - 100) / 15 * 0.3
            + offsets[mt]
            + rng.gauss(0, 0.8)
        )
        scored.append((score, [round(rpm, 1), round(temp, 1), round(vib, 2), round(pres, 1), mt]))
    threshold = sorted(s for s, _ in scored)[int(ROWS * 0.75)]
    rows = [r + [s >= threshold] for s, r in scored]
    _write("binary_failure", ["rpm", "temperature", "vibration", "pressure", "machine_type", "failure"], rows)


def regression_power_output(rng):
    rows = []
    for _ in range(ROWS):
        ws = rng.uniform(3, 25)
        rho = rng.gauss(1.2, 0.05)
        pitch = rng.uniform(0, 20)
        temp = rng.gauss(15, 10)
        # kW: cubic in wind speed, saturating at rated power, scaled by density and pitch loss
        ideal = min(0.5 * rho * 4000 * (ws ** 3) / 1000 * 0.4, 2000.0)
        power = ideal * math.cos(math.radians(pitch)) ** 2 * (1 - 0.002 * max(temp - 25, 0))
        power += rng.gauss(0, 60)
        rows.append([round(ws, 2), round(rho, 3), round(pitch, 1), round(temp, 1), round(max(power, 0), 1)])
    _write("regression_power_output", ["wind_speed", "air_density", "blade_pitch", "temperature", "power_output"], rows)


def multiclass_fault_type(rng):
    rows = []
    for _ in range(ROWS):
        vx = abs(rng.gauss(2.0, 1.0))
        vy = abs(rng.gauss(2.0, 1.0))
        temp = rng.gauss(65, 12)
        cur = rng.gauss(20, 4)
        rpm = rng.gauss(1500, 250)
        scores = {
            "none": 0.5,
            "bearing": 0.9 * (temp - 65) / 12 + 0.5 * (vx - 2.0) + rng.gauss(0, 0.6),
            "misalignment": 0.9 * (vy - 2.0) + 0.4 * (cur - 20) / 4 + rng.gauss(0, 0.6),
            "imbalance": 0.9 * (rpm - 1500) / 250 + 0.4 * (vx - 2.0) + rng.gauss(0, 0.6),
        }
        label = max(scores, key=scores.get)
        rows.append([round(vx, 2), round(vy, 2), round(temp, 1), round(cur, 1), round(rpm, 1), label])
    _write("multiclass_fault_type", ["vibration_x", "vibration_y", "temperature", "current", "rpm", "fault_type"], rows)


def main():
    rng = random.Random(SEED)
    binary_failure(rng)
    regression_power_output(rng)
    multiclass_fault_type(rng)


if __name__ == "__main__":
    main()
