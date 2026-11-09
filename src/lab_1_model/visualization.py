"""Charts for the trend, harmonic, residual, and forecast stages."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from lab_1_model.data import Observation
from lab_1_model.models import AnalysisResult


class AnalysisPlotter:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def create_plots(
        self,
        observations: tuple[Observation, ...] | list[Observation],
        result: AnalysisResult,
    ) -> tuple[Path, ...]:
        times = [observation.time for observation in observations]
        values = [observation.value for observation in observations]
        output_paths: list[Path] = []

        figure, axis = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=times, y=values, marker="o", ax=axis, label="Исходный ряд")
        axis.set(title="Исходный динамический ряд", xlabel="Время t", ylabel="Прибыль")
        output_paths.append(self._save(figure, "01 исходный ряд.png"))

        figure, axis = plt.subplots(figsize=(11, 6))
        sns.lineplot(x=times, y=values, marker="o", ax=axis, label="Исходный ряд")
        for fit in result.trend_fits:
            sns.lineplot(x=times, y=fit.values, ax=axis, label=fit.name)
        axis.set(title="Сравнение моделей тренда", xlabel="Время t", ylabel="Прибыль")
        output_paths.append(self._save(figure, "02 сравнение трендов.png"))

        detrended_values = [
            value - trend
            for value, trend in zip(values, result.quadratic_trend.values)
        ]
        figure, axis = plt.subplots(figsize=(11, 6))
        sns.lineplot(
            x=times,
            y=detrended_values,
            marker="o",
            ax=axis,
            label="Ряд после удаления квадратичного тренда",
        )
        sns.lineplot(
            x=times,
            y=result.seasonal_values,
            ax=axis,
            linewidth=2.5,
            label="Сумма двух гармоник",
        )
        for harmonic in result.fourier_fit.harmonics:
            component = [
                harmonic.cosine_coefficient
                * math.cos(harmonic.angular_frequency * (time - times[0]))
                + harmonic.sine_coefficient
                * math.sin(harmonic.angular_frequency * (time - times[0]))
                for time in times
            ]
            sns.lineplot(
                x=times,
                y=component,
                ax=axis,
                linestyle="--",
                label=f"Гармоника {harmonic.number}",
            )
        axis.set(
            title="Гармонический анализ (K0=2)",
            xlabel="Время t",
            ylabel="Отклонение от тренда",
        )
        output_paths.append(self._save(figure, "03 гармонический анализ.png"))

        figure, axis = plt.subplots(figsize=(11, 6))
        sns.lineplot(x=times, y=values, marker="o", ax=axis, label="Исходный ряд")
        sns.lineplot(
            x=times,
            y=result.quadratic_trend.values,
            ax=axis,
            linestyle="--",
            label="Квадратичный тренд",
        )
        sns.lineplot(
            x=times,
            y=result.fitted_values,
            ax=axis,
            label="Тренд + гармоники",
        )
        axis.set(title="Итоговая модель", xlabel="Время t", ylabel="Прибыль")
        output_paths.append(self._save(figure, "04 итоговая модель.png"))

        figure, axis = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            x=times,
            y=result.residuals,
            marker="o",
            ax=axis,
            label="Остаточная компонента",
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set(title="Остатки модели", xlabel="Время t", ylabel="Остаток")
        output_paths.append(self._save(figure, "05 остатки.png"))

        figure, axis = plt.subplots(figsize=(11, 6))
        sns.lineplot(x=times, y=values, marker="o", ax=axis, label="Исходные данные")
        forecast_times = [times[-1], *result.forecast_times]
        forecast_values = [values[-1], *result.forecast_values]
        sns.lineplot(
            x=forecast_times,
            y=forecast_values,
            marker="o",
            linestyle="--",
            ax=axis,
            label="Прогноз",
        )
        axis.set(
            title="Прогноз динамического ряда",
            xlabel="Время t",
            ylabel="Прибыль",
        )
        output_paths.append(self._save(figure, "06 прогноз.png"))

        return tuple(output_paths)

    def _save(self, figure: plt.Figure, filename: str) -> Path:
        destination = self.output_directory / filename
        figure.tight_layout()
        figure.savefig(destination, dpi=150)
        plt.close(figure)
        return destination
