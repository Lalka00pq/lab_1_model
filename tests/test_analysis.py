from pathlib import Path
import unittest

from lab_1_model.analysis import TimeSeriesAnalyzer
from lab_1_model.data import load_observations
from lab_1_model.trends import (
    ExponentialSmoothingTrend,
    MovingAverageTrend,
    QuadraticTrend,
)


class TrendTests(unittest.TestCase):
    def test_moving_average_uses_available_values_at_edges(self) -> None:
        fit = MovingAverageTrend(window_radius=2).fit(
            [0, 1, 2, 3, 4],
            [1, 3, 5, 7, 9],
        )

        self.assertEqual(fit.values, (2.0, 4.0, 5.0, 6.0, 8.0))

    def test_exponential_smoothing_updates_initial_level_with_first_value(self) -> None:
        fit = ExponentialSmoothingTrend().fit(
            [0, 1, 2],
            [1, 2, 3],
        )

        self.assertAlmostEqual(fit.values[0], (1 - 0.5) * 2 + 0.5 * 1)

    def test_quadratic_trend_recovers_exact_polynomial(self) -> None:
        times = [0.0, 1.0, 2.0, 3.0]
        values = [4.0 + 2.0 * time + time**2 for time in times]

        fit = QuadraticTrend().fit(times, values)

        for actual, expected in zip(fit.parameters, (4.0, 2.0, 1.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(fit.sum_squared_error, 0.0)


class CompleteAnalysisTests(unittest.TestCase):
    def test_supplied_data_produces_two_harmonics_and_required_forecast(self) -> None:
        data_path = (
            Path(__file__).parents[1]
            / "src"
            / "lab_1_model"
            / "data.csv"
        )
        observations = load_observations(data_path)
        result = TimeSeriesAnalyzer().analyze(observations)

        self.assertEqual(result.forecast_times, (64.0, 66.0, 68.0, 70.0))
        self.assertEqual(len(result.fourier_fit.harmonics), 2)
        self.assertEqual(len(result.residuals), len(observations))
        self.assertTrue(result.best_overall_model)
        self.assertEqual(result.best_trend.name, MovingAverageTrend.name)
        self.assertEqual(
            result.best_overall_model,
            "Квадратичный тренд + две гармоники",
        )
        self.assertTrue(all(map(float.is_integer, result.forecast_times)))


if __name__ == "__main__":
    unittest.main()
