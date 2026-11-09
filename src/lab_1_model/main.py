"""Command-line entry point for the dynamic-series laboratory."""

import argparse
import csv
import math
from pathlib import Path
from typing import Final

from lab_1_model.analysis import TimeSeriesAnalyzer
from lab_1_model.data import Observation, load_observations
from lab_1_model.models import AnalysisResult
from lab_1_model.visualization import AnalysisPlotter


DEFAULT_DATA_PATH: Final = Path(__file__).with_name("data.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a dynamic-series model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for charts, tables, and the analysis report.",
    )
    arguments = parser.parse_args(argv)

    observations = load_observations(arguments.data)
    result = TimeSeriesAnalyzer().analyze(observations)
    arguments.output.mkdir(parents=True, exist_ok=True)
    plot_paths = AnalysisPlotter(arguments.output / "graphs").create_plots(
        observations, result
    )
    write_results(arguments.output, observations, result)

    print(format_report(observations, result))
    print(f"\nРезультаты сохранены в: {arguments.output.resolve()}")
    print("Графики:")
    for path in plot_paths:
        print(f"  {path}")


def write_results(
    output_directory: Path,
    observations: list[Observation],
    result: AnalysisResult,
) -> None:
    trend_fits = result.trend_fits
    with (output_directory / "model_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "t",
                "Y",
                *(fit.name for fit in trend_fits),
                "V",
                "U+V",
                "E",
            ]
        )
        for index, observation in enumerate(observations):
            writer.writerow(
                [
                    observation.time,
                    observation.value,
                    *(fit.values[index] for fit in trend_fits),
                    result.seasonal_values[index],
                    result.fitted_values[index],
                    result.residuals[index],
                ]
            )

    with (output_directory / "forecast.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["t", "Прогноз"])
        writer.writerows(zip(result.forecast_times, result.forecast_values))

    (output_directory / "analysis_report.txt").write_text(
        format_report(observations, result) + "\n",
        encoding="utf-8",
    )


def format_report(
    observations: list[Observation],
    result: AnalysisResult,
) -> str:
    lines = [
        "АНАЛИЗ ДИНАМИЧЕСКОГО РЯДА",
        f"Число наблюдений: {len(observations)}",
        "",
        "Сравнение трендов по сумме квадратов ошибок R:",
    ]
    for fit in result.trend_fits:
        lines.append(
            f"  {fit.name}: R = {fit.sum_squared_error:.6f}"
            f"{'  <-- лучший тренд' if fit is result.best_trend else ''}"
        )

    constant, linear, quadratic = result.quadratic_trend.parameters
    lines.extend(
        [
            "",
            f"Выбранная модель среди трендов: {result.best_trend.name}.",
            "Для последующего гармонического анализа по методичке "
            "использован квадратичный тренд:",
            f"  U(t) = {constant:.8f} + ({linear:.8f})*t "
            f"+ ({quadratic:.8f})*t^2",
            f"Период ряда для разложения Фурье: {result.fourier_fit.period:.6f}",
            f"Средняя составляющая Фурье a0: {result.fourier_fit.intercept:.6f}",
            "Коэффициенты гармоник:",
        ]
    )
    for harmonic in result.fourier_fit.harmonics:
        lines.append(
            f"  k={harmonic.number}: "
            f"a={harmonic.cosine_coefficient:.8f}, "
            f"b={harmonic.sine_coefficient:.8f}, "
            f"omega={harmonic.angular_frequency:.8f}"
        )

    diagnostics = result.diagnostics
    lines.extend(
        [
            "",
            f"Лучшая модель по R среди трендов и итоговой модели: "
            f"{result.best_overall_model} "
            f"(R = {result.best_overall_sum_squared_error:.6f}).",
            f"Случайность по критерию пиков: "
            f"{'условие выполнено' if diagnostics.residuals_are_random else 'условие не выполнено'} "
            f"(P={diagnostics.peak_count}, ожидание={diagnostics.expected_peak_count:.3f}).",
            f"Нормальность остатков: {diagnostics.normality_status} "
            f"(асимметрия={diagnostics.skewness:.6f}, "
            f"эксцесс={diagnostics.excess_kurtosis:.6f}).",
            f"Нулевое среднее остатков по t-критерию: "
            f"{'условие выполнено' if diagnostics.residual_mean_is_zero else 'условие не выполнено'} "
            f"(t={diagnostics.mean_test_statistic:.6f}, "
            f"t критическое={diagnostics.mean_test_critical_value:.6f}, "
            f"среднее={diagnostics.residual_mean:.6f}).",
            f"Критерий Дарбина—Уотсона d: {diagnostics.durbin_watson:.6f} "
            "(для формального вывода нужны табличные границы d1 и d2).",
            "",
            "Точность итоговой модели (квадратичный тренд + две гармоники):",
            f"  Сумма квадратов ошибок: {diagnostics.sum_squared_error:.6f}",
            f"  Стандартная ошибка: {_format_number(diagnostics.standard_error)}",
            f"  Коэффициент сходимости Phi^2: {_format_number(diagnostics.convergence_coefficient)}",
            f"  Коэффициент детерминации D: {_format_number(diagnostics.determination_coefficient)}",
            f"  Индекс корреляции R: {_format_number(diagnostics.correlation_index)}",
            f"  Средняя ошибка аппроксимации: "
            f"{diagnostics.mean_absolute_percentage_error:.6f}%",
            "",
            "Прогноз итоговой модели:",
        ]
    )
    lines.extend(
        f"  t={time:g}: {value:.6f}"
        for time, value in zip(result.forecast_times, result.forecast_values)
    )
    return "\n".join(lines)


def _format_number(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "не определено"