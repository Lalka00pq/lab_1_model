"""Orchestration of the complete time-series analysis."""

from collections.abc import Sequence

from lab_1_model.diagnostics import ResidualAnalyzer
from lab_1_model.data import Observation
from lab_1_model.models import AnalysisResult
from lab_1_model.seasonality import HarmonicAnalyzer
from lab_1_model.trends import (
    ExponentialSmoothingTrend,
    MovingAverageTrend,
    QuadraticTrend,
)


class TimeSeriesAnalyzer:
    def __init__(self) -> None:
        self.trend_models = (
            MovingAverageTrend(window_radius=2),
            ExponentialSmoothingTrend(),
            QuadraticTrend(),
        )
        self.harmonic_analyzer = HarmonicAnalyzer(harmonic_count=2)
        self.residual_analyzer = ResidualAnalyzer()

    def analyze(self, observations: Sequence[Observation]) -> AnalysisResult:
        if len(observations) < 4:
            raise ValueError("At least four observations are required for analysis.")

        times = tuple(observation.time for observation in observations)
        values = tuple(observation.value for observation in observations)
        trend_fits = tuple(
            model.fit(times, values) for model in self.trend_models
        )
        best_trend = min(trend_fits, key=lambda fit: fit.sum_squared_error)
        quadratic_trend = next(
            fit
            for fit in trend_fits
            if fit.name == QuadraticTrend.name
        )

        detrended_values = tuple(
            value - trend
            for value, trend in zip(values, quadratic_trend.values)
        )
        fourier_fit = self.harmonic_analyzer.fit(times, detrended_values)
        seasonal_values = fourier_fit.predict_many(times)
        fitted_values = tuple(
            trend + season
            for trend, season in zip(quadratic_trend.values, seasonal_values)
        )
        residuals, diagnostics = self.residual_analyzer.analyze(
            values,
            fitted_values,
            parameter_count=8,
        )

        model_errors = [
            (fit.name, fit.sum_squared_error) for fit in trend_fits
        ]
        combined_model_name = "Квадратичный тренд + две гармоники"
        model_errors.append((combined_model_name, diagnostics.sum_squared_error))
        best_overall_model, best_overall_error = min(
            model_errors, key=lambda item: item[1]
        )

        step = times[1] - times[0]
        forecast_times = tuple(
            times[-1] + step * offset for offset in range(1, 5)
        )
        constant, linear, quadratic = quadratic_trend.parameters
        forecast_values = tuple(
            constant
            + linear * time
            + quadratic * time**2
            + fourier_fit.predict(time)
            for time in forecast_times
        )

        return AnalysisResult(
            trend_fits=trend_fits,
            best_trend=best_trend,
            quadratic_trend=quadratic_trend,
            best_overall_model=best_overall_model,
            best_overall_sum_squared_error=best_overall_error,
            seasonal_values=seasonal_values,
            fourier_fit=fourier_fit,
            fitted_values=fitted_values,
            residuals=residuals,
            diagnostics=diagnostics,
            forecast_times=forecast_times,
            forecast_values=forecast_values,
        )
