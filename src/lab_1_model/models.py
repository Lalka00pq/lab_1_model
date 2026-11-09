"""Results shared by the time-series modeling components."""

from dataclasses import dataclass
from math import cos, sin


@dataclass(frozen=True)
class TrendFit:
    name: str
    values: tuple[float, ...]
    parameters: tuple[float, ...]
    sum_squared_error: float


@dataclass(frozen=True)
class Harmonic:
    number: int
    cosine_coefficient: float
    sine_coefficient: float
    angular_frequency: float


@dataclass(frozen=True)
class FourierFit:
    intercept: float
    origin: float
    period: float
    harmonics: tuple[Harmonic, ...]

    def predict(self, time: float) -> float:
        phase_time = time - self.origin
        return self.intercept + sum(
            harmonic.cosine_coefficient
            * cos(harmonic.angular_frequency * phase_time)
            + harmonic.sine_coefficient
            * sin(harmonic.angular_frequency * phase_time)
            for harmonic in self.harmonics
        )

    def predict_many(self, times: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        return tuple(self.predict(time) for time in times)


@dataclass(frozen=True)
class ResidualDiagnostics:
    peak_count: int
    expected_peak_count: float
    peak_count_variance: float
    residuals_are_random: bool
    skewness: float
    excess_kurtosis: float
    normality_status: str
    residual_mean: float
    mean_test_statistic: float
    mean_test_critical_value: float
    residual_mean_is_zero: bool
    durbin_watson: float
    sum_squared_error: float
    standard_error: float
    convergence_coefficient: float
    determination_coefficient: float
    correlation_index: float
    mean_absolute_percentage_error: float


@dataclass(frozen=True)
class AnalysisResult:
    trend_fits: tuple[TrendFit, ...]
    best_trend: TrendFit
    quadratic_trend: TrendFit
    best_overall_model: str
    best_overall_sum_squared_error: float
    seasonal_values: tuple[float, ...]
    fourier_fit: FourierFit
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    diagnostics: ResidualDiagnostics
    forecast_times: tuple[float, ...]
    forecast_values: tuple[float, ...]
