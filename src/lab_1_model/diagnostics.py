"""Residual randomness, distribution, and accuracy checks."""

import math
from collections.abc import Sequence

from lab_1_model.models import ResidualDiagnostics


class ResidualAnalyzer:
    def analyze(
        self,
        actual: Sequence[float],
        fitted: Sequence[float],
        parameter_count: int,
    ) -> tuple[tuple[float, ...], ResidualDiagnostics]:
        if len(actual) != len(fitted) or len(actual) < 4:
            raise ValueError("Residual analysis requires at least four paired values.")

        residuals = tuple(value - estimate for value, estimate in zip(actual, fitted))
        count = len(residuals)
        residual_mean = sum(residuals) / count
        centered = [residual - residual_mean for residual in residuals]
        second_moment = sum(residual**2 for residual in centered) / count
        standard_deviation = math.sqrt(second_moment)
        if standard_deviation == 0:
            skewness = 0.0
            excess_kurtosis = 0.0
        else:
            skewness = (
                sum(residual**3 for residual in centered)
                / count
                / standard_deviation**3
            )
            excess_kurtosis = (
                sum(residual**4 for residual in centered)
                / count
                / standard_deviation**4
                - 3.0
            )

        skewness_error = math.sqrt(6.0 * (count - 2) / ((count + 1) * (count + 3)))
        kurtosis_error = math.sqrt(
            24.0
            * count
            * (count - 2)
            * (count - 3)
            / ((count + 1) ** 2 * (count + 3) * (count + 5))
        )
        adjusted_kurtosis = excess_kurtosis + 6.0 / (count + 1)
        normal_at_5_percent = (
            abs(skewness) < 1.5 * skewness_error
            and abs(adjusted_kurtosis) < 1.5 * kurtosis_error
        )
        definitely_non_normal = (
            abs(skewness) >= 2.0 * skewness_error
            or abs(adjusted_kurtosis) >= 2.0 * kurtosis_error
        )
        if normal_at_5_percent:
            normality_status = "гипотеза не отвергается"
        elif definitely_non_normal:
            normality_status = "нормальность не подтверждается"
        else:
            normality_status = "результат проверки неопределенный"

        peaks = sum(
            (residuals[index - 1] > residuals[index] < residuals[index + 1])
            or (residuals[index - 1] < residuals[index] > residuals[index + 1])
            for index in range(1, count - 1)
        )
        expected_peaks = 2.0 * (count - 2) / 3.0
        peak_variance = (16.0 * count - 29.0) / 90.0
        residuals_are_random = peaks > expected_peaks - 1.96 * math.sqrt(peak_variance)

        residual_sum_squares = sum(residual**2 for residual in residuals)
        sample_deviation = math.sqrt(
            sum(residual**2 for residual in centered) / (count - 1)
        )
        if sample_deviation == 0:
            mean_test_statistic = 0.0
            residual_mean_is_zero = True
        else:
            mean_test_statistic = abs(residual_mean) / (
                sample_deviation / math.sqrt(count)
            )
            residual_mean_is_zero = mean_test_statistic <= _critical_95(count - 1)

        differences = sum(
            (residuals[index] - residuals[index - 1]) ** 2 for index in range(1, count)
        )
        durbin_watson = (
            differences / residual_sum_squares if residual_sum_squares > 0 else 0.0
        )

        degrees_of_freedom = count - parameter_count
        standard_error = (
            math.sqrt(residual_sum_squares / degrees_of_freedom)
            if degrees_of_freedom > 0
            else math.nan
        )
        total_sum_squares = sum((value - sum(actual) / count) ** 2 for value in actual)
        convergence_coefficient = (
            residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else math.nan
        )
        determination_coefficient = 1.0 - convergence_coefficient
        correlation_index = (
            math.sqrt(determination_coefficient)
            if determination_coefficient >= 0
            else math.nan
        )
        mean_absolute_percentage_error = (
            100.0
            / count
            * sum(
                abs(value - estimate) / abs(value)
                for value, estimate in zip(actual, fitted)
                if value != 0
            )
        )

        diagnostics = ResidualDiagnostics(
            peaks,
            expected_peaks,
            peak_variance,
            residuals_are_random,
            skewness,
            excess_kurtosis,
            normality_status,
            residual_mean,
            mean_test_statistic,
            _critical_95(count - 1),
            residual_mean_is_zero,
            durbin_watson,
            residual_sum_squares,
            standard_error,
            convergence_coefficient,
            determination_coefficient,
            correlation_index,
            mean_absolute_percentage_error,
        )
        return residuals, diagnostics


def _critical_95(degrees_of_freedom: int) -> float:
    """Cornish-Fisher approximation to the two-sided 95% Student critical value."""
    z_value = 1.959963984540054
    degrees = float(degrees_of_freedom)
    return (
        z_value
        + (z_value**3 + z_value) / (4.0 * degrees)
        + (5.0 * z_value**5 + 16.0 * z_value**3 + 3.0 * z_value) / (96.0 * degrees**2)
        + (3.0 * z_value**7 + 19.0 * z_value**5 + 17.0 * z_value**3 - 15.0 * z_value)
        / (384.0 * degrees**3)
    )
