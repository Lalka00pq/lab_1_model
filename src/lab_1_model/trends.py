"""Trend estimation methods required by the laboratory assignment."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lab_1_model.models import TrendFit


class TrendModel(ABC):
    name: str

    @abstractmethod
    def fit(self, times: Sequence[float], values: Sequence[float]) -> TrendFit:
        """Fit this trend model to the supplied observations."""

    @staticmethod
    def _result(
        name: str,
        predicted: Sequence[float],
        actual: Sequence[float],
        parameters: Sequence[float] = (),
    ) -> TrendFit:
        error = sum((actual_value - fit_value) ** 2 for actual_value, fit_value in zip(actual, predicted))
        return TrendFit(name, tuple(predicted), tuple(parameters), error)


class MovingAverageTrend(TrendModel):
    name = "Скользящее среднее (L=2)"

    def __init__(self, window_radius: int = 2) -> None:
        if window_radius < 1:
            raise ValueError("The moving-average window radius must be positive.")
        self.window_radius = window_radius

    def fit(self, times: Sequence[float], values: Sequence[float]) -> TrendFit:
        del times
        predicted = []
        for index in range(len(values)):
            if index == 0:
                left, right = 0, min(2, len(values))
            elif index == len(values) - 1:
                left, right = max(0, len(values) - 2), len(values)
            else:
                left = max(0, index - self.window_radius)
                right = min(len(values), index + self.window_radius + 1)
            predicted.append(sum(values[left:right]) / (right - left))
        return self._result(self.name, predicted, values, (float(self.window_radius),))


class ExponentialSmoothingTrend(TrendModel):
    name = "Экспоненциальное сглаживание"

    def fit(self, times: Sequence[float], values: Sequence[float]) -> TrendFit:
        del times
        alpha = 2.0 / (len(values) + 1)
        initial_count = min(3, len(values))
        level = sum(values[:initial_count]) / initial_count
        predicted = []
        for value in values:
            level = (1.0 - alpha) * level + alpha * value
            predicted.append(level)
        return self._result(self.name, predicted, values, (alpha, float(initial_count)))


class QuadraticTrend(TrendModel):
    name = "Полиномиальный тренд второй степени"

    def fit(self, times: Sequence[float], values: Sequence[float]) -> TrendFit:
        if len(times) != len(values) or len(times) < 3:
            raise ValueError("A quadratic trend requires at least three paired observations.")

        center = (times[0] + times[-1]) / 2.0
        scale = max(abs(time - center) for time in times)
        if scale == 0:
            raise ValueError("A quadratic trend requires distinct observation times.")
        normalized_times = [(time - center) / scale for time in times]

        sums = [
            sum(time**power for time in normalized_times)
            for power in range(5)
        ]
        right_hand_side = [
            sum(value * time**power for value, time in zip(values, normalized_times))
            for power in range(3)
        ]
        normalized_coefficients = _solve_linear_system(
            [
                [sums[row + column] for column in range(3)]
                for row in range(3)
            ],
            right_hand_side,
        )

        constant, linear, quadratic = normalized_coefficients
        quadratic_original = quadratic / scale**2
        linear_original = linear / scale - 2.0 * center * quadratic_original
        constant_original = (
            constant
            - center * linear / scale
            + center**2 * quadratic_original
        )
        coefficients = (constant_original, linear_original, quadratic_original)
        predicted = [
            coefficients[0] + coefficients[1] * time + coefficients[2] * time**2
            for time in times
        ]
        return self._result(self.name, predicted, values, coefficients)


def _solve_linear_system(
    matrix: list[list[float]], vector: list[float]
) -> list[float]:
    """Solve a small dense system with Gaussian elimination and pivoting."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if math.isclose(augmented[pivot][column], 0.0, abs_tol=1e-12):
            raise ValueError("The data do not determine a unique quadratic trend.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(augmented[row], augmented[column])
            ]

    return [augmented[row][-1] for row in range(size)]
