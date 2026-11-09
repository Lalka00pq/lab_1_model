"""Input data types and CSV loading."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    index: int
    time: float
    value: float


def load_observations(path: Path) -> list[Observation]:
    """Load and validate a uniformly sampled time series from the lab CSV."""
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"ti", "Прибыль"}
        available_columns = set(reader.fieldnames or ())
        missing_columns = required_columns - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV file is missing required columns: {missing}")

        observations = [
            Observation(
                index=index,
                time=float(row["ti"]),
                value=float(row["Прибыль"]),
            )
            for index, row in enumerate(reader, start=1)
        ]

    if len(observations) < 3:
        raise ValueError("At least three observations are required.")
    if any(
        not math.isfinite(number)
        for observation in observations
        for number in (observation.time, observation.value)
    ):
        raise ValueError("Observation times and values must be finite numbers.")

    differences = [
        observations[index].time - observations[index - 1].time
        for index in range(1, len(observations))
    ]
    if any(step <= 0 for step in differences):
        raise ValueError("Observation times must be strictly increasing.")
    first_step = differences[0]
    if any(not math.isclose(step, first_step) for step in differences[1:]):
        raise ValueError("Observation times must be equally spaced.")
    return observations
