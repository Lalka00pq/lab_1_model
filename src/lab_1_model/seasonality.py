"""Discrete Fourier analysis for the seasonal component."""

import math
from collections.abc import Sequence

from lab_1_model.models import FourierFit, Harmonic


class HarmonicAnalyzer:
    def __init__(self, harmonic_count: int = 2) -> None:
        if harmonic_count < 1:
            raise ValueError("At least one harmonic must be requested.")
        self.harmonic_count = harmonic_count

    def fit(self, times: Sequence[float], values: Sequence[float]) -> FourierFit:
        if len(times) != len(values) or len(times) < 3:
            raise ValueError("Fourier analysis requires at least three paired observations.")
        period = times[-1] - times[0]
        if period <= 0:
            raise ValueError("Fourier analysis requires a positive time span.")

        intercept = sum(values) / len(values)
        harmonics = []
        for number in range(1, self.harmonic_count + 1):
            angular_frequency = 2.0 * math.pi * number / period
            cosine_coefficient = (
                2.0
                / len(values)
                * sum(
                    value * math.cos(angular_frequency * (time - times[0]))
                    for time, value in zip(times, values)
                )
            )
            sine_coefficient = (
                2.0
                / len(values)
                * sum(
                    value * math.sin(angular_frequency * (time - times[0]))
                    for time, value in zip(times, values)
                )
            )
            harmonics.append(
                Harmonic(
                    number,
                    cosine_coefficient,
                    sine_coefficient,
                    angular_frequency,
                )
            )

        return FourierFit(
            intercept,
            times[0],
            period,
            tuple(harmonics),
        )
