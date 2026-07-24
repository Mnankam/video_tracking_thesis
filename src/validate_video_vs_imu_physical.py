#!/usr/bin/env python3
"""
validate_video_vs_imu_physical.py
=================================

Repository-spezifische physikalische Validierung von Lucas-Kanade-Video-
Tracking gegen IMU-Messungen für einen Ausschwingversuch.

Die Datei erweitert ``validate_video_vs_imu_v3.py``. Zuerst wird die bestehende
Video-IMU-Pipeline für alle gewünschten IMU-Achsen ausgeführt oder ein bereits
vorhandener v3-Ausgabeordner wiederverwendet. Anschließend werden aus den
synchronisierten Signalen physikalische Merkmale des freien Ausschwingens
bestimmt.

Ausgewertete physikalische Kriterien
------------------------------------
1. Abklinghüllkurve mittels Hilbert-Transformation.
2. Exponentieller Fit ``A(t) = A0 * exp(-alpha * t)``.
3. Abklingkonstante ``alpha`` in 1/s und Zeitkonstante ``tau = 1/alpha``.
4. Logarithmisches Dekrement aus aufeinanderfolgenden Maxima.
5. Gedämpfte Eigenfrequenz aus Peak-Abständen.
6. Ungedämpfte Eigenkreisfrequenz und Eigenfrequenz.
7. Dämpfungsgrad ``zeta``.
8. Vergleich der normalisierten Hüllkurven.
9. Residuen der Hüllkurven und einfache Autokorrelationsdiagnostik.
10. Vergleichs- und Ranking-Tabelle über alle IMU-Achsen.

Wichtige physikalische Einordnung
---------------------------------
Das Videosignal beschreibt eine Position, während ``linx/liny/linz`` eine
Beschleunigung und ``rotx/roty/rotz`` eine Rotationsgröße darstellen. Direkte
Amplitude, RMSE und punktweise Pearson-Korrelation sind daher nicht ohne
Weiteres physikalisch interpretierbar. Bei einem näherungsweise linearen,
unterdämpften Einmassenschwinger besitzen Position und Beschleunigung jedoch
dieselbe exponentielle Abklingrate und dieselbe Schwingungsfrequenz. Genau
diese größenunabhängigen Merkmale stehen hier im Mittelpunkt.

Die automatische Erkennung des Ausschwingfensters ist eine praktische Hilfe,
aber kein Ersatz für die Dokumentation des Versuchsablaufs. Für die endgültige
Auswertung sollten ``--decay-start-s`` und ``--decay-end-s`` anhand des
Videos, der IMU-Rohdaten und des bekannten Zeitpunkts des Ausschwingversuchs
kontrolliert bzw. explizit gesetzt werden.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import logging
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, filtfilt, find_peaks, hilbert
    from scipy.stats import linregress
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "SciPy wird für die physikalische Ausschwinganalyse benötigt. "
        "Installieren Sie die Repository-Abhängigkeiten mit "
        "`pip install -r requirements.txt`."
    ) from exc

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from validate_video_vs_imu_v3 import (  # noqa: E402
    BatchConfig,
    DEFAULT_AXES,
    run_axis_comparison,
)

LOGGER = logging.getLogger("validate_video_vs_imu_physical")


@dataclass(slots=True)
class PhysicalConfig:
    """Konfiguration der physikalischen Ausschwinganalyse."""

    output_dir: Path
    axes: tuple[str, ...] = DEFAULT_AXES
    video_name: str = "video"
    imu_name_prefix: str = "imu"
    decay_start_s: Optional[float] = None
    decay_end_s: Optional[float] = None
    auto_decay_start: bool = True
    auto_start_fraction: float = 0.98
    minimum_decay_duration_s: float = 5.0
    envelope_smoothing_s: float = 0.20
    signal_lowpass_hz: Optional[float] = None
    minimum_peak_distance_s: Optional[float] = None
    peak_prominence_fraction: float = 0.05
    minimum_peaks: int = 6
    fit_lower_envelope_fraction: float = 0.08
    fit_upper_envelope_fraction: float = 0.98
    create_plots: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self.axes = tuple(dict.fromkeys(self.axes))
        if not self.axes:
            raise ValueError("Mindestens eine IMU-Achse muss angegeben werden.")
        if self.decay_start_s is not None and self.decay_end_s is not None:
            if self.decay_end_s <= self.decay_start_s:
                raise ValueError("--decay-end-s muss größer als --decay-start-s sein.")
        if self.minimum_decay_duration_s <= 0:
            raise ValueError("minimum_decay_duration_s muss positiv sein.")
        if self.envelope_smoothing_s < 0:
            raise ValueError("envelope_smoothing_s darf nicht negativ sein.")
        if self.minimum_peaks < 3:
            raise ValueError("Für eine robuste Analyse werden mindestens 3 Peaks benötigt.")


@dataclass(slots=True)
class OscillationMetrics:
    """Physikalische Kenngrößen eines einzelnen Ausschwing-Signals."""

    signal_name: str
    status: str
    samples: int = 0
    duration_s: Optional[float] = None
    sample_rate_hz: Optional[float] = None
    decay_start_s: Optional[float] = None
    decay_end_s: Optional[float] = None
    envelope_initial: Optional[float] = None
    envelope_final: Optional[float] = None
    envelope_decay_ratio: Optional[float] = None
    alpha_1_per_s: Optional[float] = None
    tau_s: Optional[float] = None
    exponential_fit_r_squared: Optional[float] = None
    logarithmic_decrement: Optional[float] = None
    logarithmic_decrement_std: Optional[float] = None
    damped_frequency_hz: Optional[float] = None
    damped_frequency_std_hz: Optional[float] = None
    natural_frequency_hz: Optional[float] = None
    damping_ratio: Optional[float] = None
    peak_count: int = 0
    is_decaying: Optional[bool] = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class AxisPhysicalResult:
    """Vergleich der physikalischen Merkmale von Video und einer IMU-Achse."""

    axis: str
    status: str
    aligned_csv: str
    video: dict[str, Any] = field(default_factory=dict)
    imu: dict[str, Any] = field(default_factory=dict)
    alpha_relative_error_percent: Optional[float] = None
    tau_relative_error_percent: Optional[float] = None
    damped_frequency_relative_error_percent: Optional[float] = None
    natural_frequency_relative_error_percent: Optional[float] = None
    damping_ratio_absolute_error: Optional[float] = None
    damping_ratio_relative_error_percent: Optional[float] = None
    envelope_correlation: Optional[float] = None
    envelope_rmse: Optional[float] = None
    envelope_mae: Optional[float] = None
    envelope_residual_lag1_autocorrelation: Optional[float] = None
    physical_similarity_score: Optional[float] = None
    classification: str = ""
    message: str = ""
    figure_paths: list[str] = field(default_factory=list)

    def flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "axis": self.axis,
            "status": self.status,
            "aligned_csv": self.aligned_csv,
            "alpha_relative_error_percent": self.alpha_relative_error_percent,
            "tau_relative_error_percent": self.tau_relative_error_percent,
            "damped_frequency_relative_error_percent": self.damped_frequency_relative_error_percent,
            "natural_frequency_relative_error_percent": self.natural_frequency_relative_error_percent,
            "damping_ratio_absolute_error": self.damping_ratio_absolute_error,
            "damping_ratio_relative_error_percent": self.damping_ratio_relative_error_percent,
            "envelope_correlation": self.envelope_correlation,
            "envelope_rmse": self.envelope_rmse,
            "envelope_mae": self.envelope_mae,
            "envelope_residual_lag1_autocorrelation": self.envelope_residual_lag1_autocorrelation,
            "physical_similarity_score": self.physical_similarity_score,
            "classification": self.classification,
            "message": self.message,
            "figure_paths": self.figure_paths,
        }
        for prefix, values in (("video", self.video), ("imu", self.imu)):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        return row


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _relative_error_percent(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None or abs(reference) <= np.finfo(float).eps:
        return None
    return float(100.0 * abs(candidate - reference) / abs(reference))


def _safe_corr(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 3:
        return None
    first, second = first[mask], second[mask]
    if np.std(first) <= np.finfo(float).eps or np.std(second) <= np.finfo(float).eps:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _sample_rate(time_s: np.ndarray) -> float:
    differences = np.diff(np.asarray(time_s, dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        raise ValueError("Keine gültigen positiven Zeitabstände vorhanden.")
    return float(1.0 / np.median(differences))


def _uniform_resample(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpoliert ein Signal auf eine äquidistante Zeitachse."""

    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(values)
    time_s, values = time_s[mask], values[mask]
    order = np.argsort(time_s, kind="stable")
    time_s, values = time_s[order], values[order]
    unique_time, unique_indices = np.unique(time_s, return_index=True)
    time_s, values = unique_time, values[unique_indices]
    if len(time_s) < 8:
        raise ValueError("Für die Ausschwinganalyse sind mindestens 8 Werte erforderlich.")
    fs = _sample_rate(time_s)
    dt = 1.0 / fs
    count = max(8, int(math.floor((time_s[-1] - time_s[0]) / dt)) + 1)
    uniform_time = time_s[0] + np.arange(count, dtype=float) * dt
    uniform_time = uniform_time[uniform_time <= time_s[-1] + 0.25 * dt]
    uniform_values = np.interp(uniform_time, time_s, values)
    return uniform_time, uniform_values, fs


def _lowpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: Optional[float]) -> np.ndarray:
    if cutoff_hz is None:
        return np.asarray(values, dtype=float)
    nyquist = 0.5 * sample_rate_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError(
            f"Ungültige Grenzfrequenz {cutoff_hz:g} Hz für Nyquist {nyquist:g} Hz."
        )
    b, a = butter(4, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, np.asarray(values, dtype=float))


def _smooth_envelope(envelope: np.ndarray, sample_rate_hz: float, smoothing_s: float) -> np.ndarray:
    if smoothing_s <= 0:
        return np.asarray(envelope, dtype=float)
    sigma = max(1.0, smoothing_s * sample_rate_hz / 4.0)
    return gaussian_filter1d(np.asarray(envelope, dtype=float), sigma=sigma, mode="nearest")


def _analytic_envelope(values: np.ndarray, sample_rate_hz: float, smoothing_s: float) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    envelope = np.abs(hilbert(centered))
    return _smooth_envelope(envelope, sample_rate_hz, smoothing_s)


def _automatic_decay_start(
    time_s: np.ndarray,
    envelope: np.ndarray,
    fraction: float,
    minimum_duration_s: float,
) -> float:
    """Schätzt den Beginn des freien Abklingens konservativ.

    Die Methode sucht das globale Hüllkurvenmaximum und danach den ersten Punkt,
    an dem die geglättete Hüllkurve unter ``fraction * maximum`` liegt. Der
    gefundene Start wird so begrenzt, dass mindestens die geforderte
    Auswertedauer verbleibt.
    """

    if len(time_s) < 8:
        return float(time_s[0])
    max_index = int(np.nanargmax(envelope))
    maximum = float(envelope[max_index])
    threshold = max(fraction * maximum, np.finfo(float).eps)
    candidates = np.flatnonzero(
        (np.arange(len(envelope)) >= max_index) & (envelope <= threshold)
    )
    start_index = int(candidates[0]) if candidates.size else max_index
    latest_start = float(time_s[-1] - minimum_duration_s)
    return float(min(time_s[start_index], latest_start))


def _select_decay_window(
    time_s: np.ndarray,
    values: np.ndarray,
    cfg: PhysicalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    uniform_time, uniform_values, fs = _uniform_resample(time_s, values)
    filtered = _lowpass(uniform_values, fs, cfg.signal_lowpass_hz)
    filtered = filtered - np.mean(filtered)
    envelope = _analytic_envelope(filtered, fs, cfg.envelope_smoothing_s)

    if cfg.decay_start_s is not None:
        start_s = float(cfg.decay_start_s)
    elif cfg.auto_decay_start:
        start_s = _automatic_decay_start(
            uniform_time,
            envelope,
            cfg.auto_start_fraction,
            cfg.minimum_decay_duration_s,
        )
    else:
        start_s = float(uniform_time[0])

    end_s = float(cfg.decay_end_s) if cfg.decay_end_s is not None else float(uniform_time[-1])
    mask = (uniform_time >= start_s) & (uniform_time <= end_s)
    if mask.sum() < max(16, cfg.minimum_peaks * 3):
        raise ValueError(
            f"Das gewählte Ausschwingfenster [{start_s:.3f}, {end_s:.3f}] s "
            "enthält zu wenige Daten."
        )
    selected_time = uniform_time[mask]
    selected_values = filtered[mask]
    selected_envelope = envelope[mask]
    if selected_time[-1] - selected_time[0] < cfg.minimum_decay_duration_s:
        raise ValueError(
            f"Das Ausschwingfenster ist nur {selected_time[-1] - selected_time[0]:.3f} s lang; "
            f"gefordert sind mindestens {cfg.minimum_decay_duration_s:.3f} s."
        )
    return selected_time, selected_values, selected_envelope, fs, start_s, end_s


def _fit_exponential_decay(
    time_s: np.ndarray,
    envelope: np.ndarray,
    lower_fraction: float,
    upper_fraction: float,
) -> dict[str, Optional[float]]:
    """Fit der logarithmierten Hüllkurve.

    ``log(A(t)) = log(A0) - alpha * t``.
    """

    time_s = np.asarray(time_s, dtype=float)
    envelope = np.asarray(envelope, dtype=float)
    positive = envelope[np.isfinite(envelope) & (envelope > 0)]
    if positive.size < 8:
        return {}
    reference = float(np.nanmax(positive))
    lower = max(lower_fraction * reference, np.finfo(float).eps)
    upper = max(upper_fraction * reference, lower)
    mask = (
        np.isfinite(time_s)
        & np.isfinite(envelope)
        & (envelope >= lower)
        & (envelope <= upper)
    )
    if mask.sum() < 8:
        mask = np.isfinite(time_s) & np.isfinite(envelope) & (envelope > 0)
    if mask.sum() < 8:
        return {}

    relative_time = time_s[mask] - time_s[mask][0]
    log_envelope = np.log(envelope[mask])
    fit = linregress(relative_time, log_envelope)
    alpha = -float(fit.slope)
    tau = 1.0 / alpha if alpha > np.finfo(float).eps else None
    predicted = fit.intercept + fit.slope * relative_time
    residual = log_envelope - predicted
    total = float(np.sum((log_envelope - np.mean(log_envelope)) ** 2))
    residual_sum = float(np.sum(residual**2))
    r_squared = 1.0 - residual_sum / total if total > 0 else None
    return {
        "alpha_1_per_s": alpha,
        "tau_s": tau,
        "exponential_fit_r_squared": r_squared,
        "fit_intercept": float(fit.intercept),
        "fit_slope": float(fit.slope),
    }


def _estimate_peak_metrics(
    time_s: np.ndarray,
    values: np.ndarray,
    envelope: np.ndarray,
    cfg: PhysicalConfig,
) -> dict[str, Any]:
    """Bestimmt Peak-Abstände, logarithmisches Dekrement und Dämpfung."""

    duration = float(time_s[-1] - time_s[0])
    preliminary_frequency = _dominant_frequency(time_s, values)
    if cfg.minimum_peak_distance_s is not None:
        minimum_distance_s = cfg.minimum_peak_distance_s
    elif preliminary_frequency is not None and preliminary_frequency > 0:
        minimum_distance_s = max(0.25 / preliminary_frequency, 1.0 / _sample_rate(time_s))
    else:
        minimum_distance_s = max(duration / 200.0, 1.0 / _sample_rate(time_s))

    fs = _sample_rate(time_s)
    minimum_distance_samples = max(1, int(round(minimum_distance_s * fs)))
    prominence = max(
        cfg.peak_prominence_fraction * float(np.nanmax(envelope)),
        np.finfo(float).eps,
    )

    # Für das logarithmische Dekrement werden gleichgerichtete positive Maxima
    # verwendet. Ein ggf. invertiertes Videosignal bleibt dadurch unkritisch.
    positive_peaks, _ = find_peaks(
        values,
        distance=minimum_distance_samples,
        prominence=prominence,
    )
    negative_peaks, _ = find_peaks(
        -values,
        distance=minimum_distance_samples,
        prominence=prominence,
    )
    peak_indices = positive_peaks if len(positive_peaks) >= len(negative_peaks) else negative_peaks
    peak_amplitudes = envelope[peak_indices]

    valid = np.isfinite(peak_amplitudes) & (peak_amplitudes > np.finfo(float).eps)
    peak_indices = peak_indices[valid]
    peak_amplitudes = peak_amplitudes[valid]

    result: dict[str, Any] = {
        "peak_count": int(len(peak_indices)),
        "peak_indices": peak_indices,
        "peak_times": time_s[peak_indices] if len(peak_indices) else np.array([], dtype=float),
        "peak_amplitudes": peak_amplitudes,
    }
    if len(peak_indices) < 3:
        result["message"] = "Zu wenige gleichgerichtete Peaks für Dekrement und Frequenz."
        return result

    periods = np.diff(time_s[peak_indices])
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size:
        frequencies = 1.0 / periods
        result["damped_frequency_hz"] = float(np.median(frequencies))
        result["damped_frequency_std_hz"] = float(np.std(frequencies))

    decrements = np.log(peak_amplitudes[:-1] / peak_amplitudes[1:])
    decrements = decrements[np.isfinite(decrements)]
    if decrements.size:
        # Der Median ist robuster gegen einzelne fehlerhafte Tracking-Peaks.
        logarithmic_decrement = float(np.median(decrements))
        result["logarithmic_decrement"] = logarithmic_decrement
        result["logarithmic_decrement_std"] = float(np.std(decrements))
        if logarithmic_decrement > 0:
            zeta = logarithmic_decrement / math.sqrt(
                (2.0 * math.pi) ** 2 + logarithmic_decrement**2
            )
            result["damping_ratio"] = float(zeta)
            damped_frequency = _finite_float(result.get("damped_frequency_hz"))
            if damped_frequency is not None and zeta < 1.0:
                result["natural_frequency_hz"] = float(
                    damped_frequency / math.sqrt(max(1.0 - zeta**2, np.finfo(float).eps))
                )
    return result


def _dominant_frequency(time_s: np.ndarray, values: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(values) < 8:
        return None
    dt = float(np.median(np.diff(time_s)))
    if not math.isfinite(dt) or dt <= 0:
        return None
    centered = values - np.mean(values)
    window = np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(centered * window))
    frequencies = np.fft.rfftfreq(len(centered), d=dt)
    if len(spectrum) <= 1:
        return None
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return _finite_float(frequencies[index])


def analyze_oscillation(
    signal_name: str,
    time_s: np.ndarray,
    values: np.ndarray,
    cfg: PhysicalConfig,
) -> tuple[OscillationMetrics, dict[str, np.ndarray | float | dict[str, Any]]]:
    """Analysiert ein einzelnes Signal im Ausschwingfenster."""

    try:
        selected_time, selected_values, envelope, fs, start_s, end_s = _select_decay_window(
            time_s,
            values,
            cfg,
        )
        exponential = _fit_exponential_decay(
            selected_time,
            envelope,
            cfg.fit_lower_envelope_fraction,
            cfg.fit_upper_envelope_fraction,
        )
        peaks = _estimate_peak_metrics(selected_time, selected_values, envelope, cfg)
        initial_count = max(3, int(round(0.05 * len(envelope))))
        final_count = initial_count
        envelope_initial = float(np.median(envelope[:initial_count]))
        envelope_final = float(np.median(envelope[-final_count:]))
        decay_ratio = (
            envelope_final / envelope_initial
            if envelope_initial > np.finfo(float).eps
            else None
        )
        alpha = _finite_float(exponential.get("alpha_1_per_s"))
        fit_r2 = _finite_float(exponential.get("exponential_fit_r_squared"))
        is_decaying = bool(
            alpha is not None
            and alpha > 0
            and decay_ratio is not None
            and decay_ratio < 1.0
        )
        status = "success" if len(peaks.get("peak_indices", [])) >= cfg.minimum_peaks else "limited"
        message = ""
        if status == "limited":
            message = (
                f"Nur {len(peaks.get('peak_indices', []))} Peaks erkannt; "
                "Dämpfungs- und Frequenzwerte vorsichtig interpretieren."
            )
        metrics = OscillationMetrics(
            signal_name=signal_name,
            status=status,
            samples=int(len(selected_time)),
            duration_s=float(selected_time[-1] - selected_time[0]),
            sample_rate_hz=fs,
            decay_start_s=start_s,
            decay_end_s=end_s,
            envelope_initial=envelope_initial,
            envelope_final=envelope_final,
            envelope_decay_ratio=_finite_float(decay_ratio),
            alpha_1_per_s=alpha,
            tau_s=_finite_float(exponential.get("tau_s")),
            exponential_fit_r_squared=fit_r2,
            logarithmic_decrement=_finite_float(peaks.get("logarithmic_decrement")),
            logarithmic_decrement_std=_finite_float(peaks.get("logarithmic_decrement_std")),
            damped_frequency_hz=_finite_float(peaks.get("damped_frequency_hz")),
            damped_frequency_std_hz=_finite_float(peaks.get("damped_frequency_std_hz")),
            natural_frequency_hz=_finite_float(peaks.get("natural_frequency_hz")),
            damping_ratio=_finite_float(peaks.get("damping_ratio")),
            peak_count=int(peaks.get("peak_count", 0)),
            is_decaying=is_decaying,
            message=message,
        )
        details: dict[str, np.ndarray | float | dict[str, Any]] = {
            "time_s": selected_time,
            "values": selected_values,
            "envelope": envelope,
            "peaks": peaks,
            "exponential": exponential,
        }
        return metrics, details
    except Exception as exc:
        return (
            OscillationMetrics(
                signal_name=signal_name,
                status="failed",
                message=str(exc),
            ),
            {},
        )


def _normalized_envelope(time_s: np.ndarray, envelope: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    interpolated = np.interp(common_time, time_s, envelope)
    initial_count = max(3, int(round(0.05 * len(interpolated))))
    scale = float(np.median(interpolated[:initial_count]))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.nanmax(np.abs(interpolated)))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.full_like(interpolated, np.nan)
    return interpolated / scale


def _physical_score(
    alpha_error: Optional[float],
    frequency_error: Optional[float],
    envelope_correlation: Optional[float],
    video_fit_r2: Optional[float],
    imu_fit_r2: Optional[float],
) -> Optional[float]:
    """Heuristischer Ranking-Score zwischen 0 und 1.

    Der Score ist nur eine Sortierhilfe. Die einzelnen physikalischen
    Kenngrößen müssen separat berichtet werden.
    """

    components: list[tuple[float, float]] = []
    if alpha_error is not None:
        components.append((0.35, math.exp(-max(alpha_error, 0.0) / 25.0)))
    if frequency_error is not None:
        components.append((0.30, math.exp(-max(frequency_error, 0.0) / 10.0)))
    if envelope_correlation is not None:
        components.append((0.20, max(min(envelope_correlation, 1.0), -1.0) * 0.5 + 0.5))
    fit_values = [
        value for value in (video_fit_r2, imu_fit_r2)
        if value is not None and math.isfinite(value)
    ]
    if fit_values:
        components.append((0.15, max(0.0, min(1.0, float(np.mean(fit_values))))))
    if not components:
        return None
    total_weight = sum(weight for weight, _ in components)
    return float(sum(weight * value for weight, value in components) / total_weight)


def _classification(
    video: OscillationMetrics,
    imu: OscillationMetrics,
    alpha_error: Optional[float],
    frequency_error: Optional[float],
    envelope_correlation: Optional[float],
) -> tuple[str, str]:
    if video.status == "failed" or imu.status == "failed":
        return "analysis_failed", "Mindestens ein Signal konnte nicht physikalisch ausgewertet werden."
    if not video.is_decaying or not imu.is_decaying:
        return (
            "decay_not_confirmed",
            "Ein exponentiell abklingendes Verhalten wurde nicht in beiden Signalen bestätigt.",
        )
    if alpha_error is None or frequency_error is None:
        return (
            "insufficient_metrics",
            "Für eine belastbare Gegenüberstellung fehlen Dämpfungs- oder Frequenzwerte.",
        )
    if alpha_error <= 10.0 and frequency_error <= 5.0 and (envelope_correlation or -1.0) >= 0.8:
        return (
            "strong_physical_agreement",
            "Abklingrate, Eigenfrequenz und Hüllkurven stimmen stark überein.",
        )
    if alpha_error <= 25.0 and frequency_error <= 10.0 and (envelope_correlation or -1.0) >= 0.5:
        return (
            "moderate_physical_agreement",
            "Die physikalischen Ausschwingmerkmale stimmen in brauchbarer Näherung überein.",
        )
    if alpha_error <= 50.0 and frequency_error <= 20.0:
        return (
            "weak_physical_agreement",
            "Es besteht nur eine schwache Übereinstimmung der Ausschwingmerkmale.",
        )
    return (
        "no_physical_agreement",
        "Abklingrate und/oder Eigenfrequenz stimmen nicht ausreichend überein.",
    )


def _create_axis_plots(
    axis: str,
    output_dir: Path,
    video_details: dict[str, Any],
    imu_details: dict[str, Any],
    result: AxisPhysicalResult,
) -> list[str]:
    if not video_details or not imu_details:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("Matplotlib fehlt; keine physikalischen Abbildungen.")
        return []

    figure_dir = output_dir / axis / "physical" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    video_t = np.asarray(video_details["time_s"], dtype=float)
    imu_t = np.asarray(imu_details["time_s"], dtype=float)
    common_start = max(video_t[0], imu_t[0])
    common_end = min(video_t[-1], imu_t[-1])
    common_count = max(50, min(len(video_t), len(imu_t)))
    common_t = np.linspace(common_start, common_end, common_count)

    video_env = _normalized_envelope(
        video_t,
        np.asarray(video_details["envelope"], dtype=float),
        common_t,
    )
    imu_env = _normalized_envelope(
        imu_t,
        np.asarray(imu_details["envelope"], dtype=float),
        common_t,
    )

    path = figure_dir / "decay_envelopes.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(common_t, video_env, label="Video-Hüllkurve")
    ax.plot(common_t, imu_env, label=f"IMU {axis}-Hüllkurve")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Normierte Hüllkurve [-]")
    ax.set_title(f"Abklinghüllkurven: Video vs. IMU {axis}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    path = figure_dir / "log_envelope_fit.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, details in (("Video", video_details), (f"IMU {axis}", imu_details)):
        t = np.asarray(details["time_s"], dtype=float)
        env = np.asarray(details["envelope"], dtype=float)
        normalized = env / max(float(np.median(env[:max(3, int(0.05 * len(env)))])), np.finfo(float).eps)
        ax.plot(t, np.log(np.maximum(normalized, np.finfo(float).eps)), label=f"{label}: log(Hüllkurve)")
        exponential = details.get("exponential", {})
        slope = _finite_float(exponential.get("fit_slope"))
        intercept = _finite_float(exponential.get("fit_intercept"))
        if slope is not None and intercept is not None:
            rel_t = t - t[0]
            fitted = intercept + slope * rel_t
            fitted -= fitted[0]
            ax.plot(t, fitted, linestyle="--", label=f"{label}: Exponentialfit")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("log(normierte Amplitude)")
    ax.set_title(f"Exponentialer Abklingfit: Video vs. IMU {axis}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    residual = video_env - imu_env
    path = figure_dir / "envelope_residual.png"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(common_t, residual)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Video-Hüllkurve − IMU-Hüllkurve")
    ax.set_title(f"Residuum der normierten Hüllkurven: {axis}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    path = figure_dir / "signal_with_peaks.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, details in (("Video", video_details), (f"IMU {axis}", imu_details)):
        t = np.asarray(details["time_s"], dtype=float)
        values = np.asarray(details["values"], dtype=float)
        scale = float(np.std(values))
        plotted = values / scale if scale > np.finfo(float).eps else values
        ax.plot(t, plotted, alpha=0.75, label=label)
        peak_times = np.asarray(details.get("peaks", {}).get("peak_times", []), dtype=float)
        if len(peak_times):
            peak_values = np.interp(peak_times, t, plotted)
            ax.scatter(peak_times, peak_values, s=18)
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Standardisierte Amplitude [-]")
    ax.set_title(f"Erkannte Ausschwing-Peaks: Video vs. IMU {axis}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def analyze_axis(
    axis: str,
    aligned_csv: Path,
    cfg: PhysicalConfig,
) -> AxisPhysicalResult:
    if not aligned_csv.is_file():
        return AxisPhysicalResult(
            axis=axis,
            status="missing",
            aligned_csv=str(aligned_csv),
            message="aligned_signals.csv fehlt. Zuerst v3 ausführen oder --run-base-validation verwenden.",
        )

    frame = pd.read_csv(aligned_csv)
    if "time_s" not in frame.columns:
        return AxisPhysicalResult(
            axis=axis,
            status="failed",
            aligned_csv=str(aligned_csv),
            message="Spalte 'time_s' fehlt in aligned_signals.csv.",
        )
    signal_columns = [column for column in frame.columns if column not in {"time_s", "residual"}]
    video_column = cfg.video_name if cfg.video_name in frame.columns else (signal_columns[0] if signal_columns else None)
    expected_imu = f"{cfg.imu_name_prefix} {axis}"
    imu_column = expected_imu if expected_imu in frame.columns else (
        signal_columns[1] if len(signal_columns) >= 2 else None
    )
    if video_column is None or imu_column is None:
        return AxisPhysicalResult(
            axis=axis,
            status="failed",
            aligned_csv=str(aligned_csv),
            message=f"Video-/IMU-Spalten konnten nicht erkannt werden. Spalten: {list(frame.columns)}",
        )

    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    video_values = pd.to_numeric(frame[video_column], errors="coerce").to_numpy(float)
    imu_values = pd.to_numeric(frame[imu_column], errors="coerce").to_numpy(float)

    video_metrics, video_details = analyze_oscillation(video_column, time_s, video_values, cfg)
    imu_metrics, imu_details = analyze_oscillation(imu_column, time_s, imu_values, cfg)

    alpha_error = _relative_error_percent(video_metrics.alpha_1_per_s, imu_metrics.alpha_1_per_s)
    tau_error = _relative_error_percent(video_metrics.tau_s, imu_metrics.tau_s)
    damped_frequency_error = _relative_error_percent(
        video_metrics.damped_frequency_hz,
        imu_metrics.damped_frequency_hz,
    )
    natural_frequency_error = _relative_error_percent(
        video_metrics.natural_frequency_hz,
        imu_metrics.natural_frequency_hz,
    )
    damping_ratio_absolute_error = (
        abs(video_metrics.damping_ratio - imu_metrics.damping_ratio)
        if video_metrics.damping_ratio is not None and imu_metrics.damping_ratio is not None
        else None
    )
    damping_ratio_relative_error = _relative_error_percent(
        video_metrics.damping_ratio,
        imu_metrics.damping_ratio,
    )

    envelope_correlation = envelope_rmse = envelope_mae = residual_lag1 = None
    if video_details and imu_details:
        video_t = np.asarray(video_details["time_s"], dtype=float)
        imu_t = np.asarray(imu_details["time_s"], dtype=float)
        common_start = max(video_t[0], imu_t[0])
        common_end = min(video_t[-1], imu_t[-1])
        if common_end > common_start:
            count = max(50, min(len(video_t), len(imu_t)))
            common_t = np.linspace(common_start, common_end, count)
            video_env = _normalized_envelope(
                video_t,
                np.asarray(video_details["envelope"], dtype=float),
                common_t,
            )
            imu_env = _normalized_envelope(
                imu_t,
                np.asarray(imu_details["envelope"], dtype=float),
                common_t,
            )
            residual = video_env - imu_env
            envelope_correlation = _safe_corr(video_env, imu_env)
            envelope_rmse = float(np.sqrt(np.nanmean(residual**2)))
            envelope_mae = float(np.nanmean(np.abs(residual)))
            residual_lag1 = _safe_corr(residual[:-1], residual[1:]) if len(residual) >= 4 else None

    classification, message = _classification(
        video_metrics,
        imu_metrics,
        alpha_error,
        natural_frequency_error if natural_frequency_error is not None else damped_frequency_error,
        envelope_correlation,
    )
    score = _physical_score(
        alpha_error,
        natural_frequency_error if natural_frequency_error is not None else damped_frequency_error,
        envelope_correlation,
        video_metrics.exponential_fit_r_squared,
        imu_metrics.exponential_fit_r_squared,
    )

    result = AxisPhysicalResult(
        axis=axis,
        status="success" if video_metrics.status != "failed" and imu_metrics.status != "failed" else "failed",
        aligned_csv=str(aligned_csv),
        video=video_metrics.as_dict(),
        imu=imu_metrics.as_dict(),
        alpha_relative_error_percent=alpha_error,
        tau_relative_error_percent=tau_error,
        damped_frequency_relative_error_percent=damped_frequency_error,
        natural_frequency_relative_error_percent=natural_frequency_error,
        damping_ratio_absolute_error=_finite_float(damping_ratio_absolute_error),
        damping_ratio_relative_error_percent=damping_ratio_relative_error,
        envelope_correlation=envelope_correlation,
        envelope_rmse=envelope_rmse,
        envelope_mae=envelope_mae,
        envelope_residual_lag1_autocorrelation=residual_lag1,
        physical_similarity_score=score,
        classification=classification,
        message=message,
    )
    if cfg.create_plots:
        result.figure_paths = _create_axis_plots(
            axis,
            cfg.output_dir,
            video_details,
            imu_details,
            result,
        )
    return result


def _comparison_frame(results: list[AxisPhysicalResult]) -> pd.DataFrame:
    frame = pd.DataFrame([result.flat_dict() for result in results])
    if frame.empty:
        return frame
    status_rank = frame["status"].eq("success").astype(int)
    score = pd.to_numeric(frame["physical_similarity_score"], errors="coerce").fillna(-np.inf)
    alpha_error = pd.to_numeric(frame["alpha_relative_error_percent"], errors="coerce").fillna(np.inf)
    frequency_error = pd.to_numeric(
        frame["natural_frequency_relative_error_percent"],
        errors="coerce",
    )
    frequency_error = frequency_error.fillna(
        pd.to_numeric(frame["damped_frequency_relative_error_percent"], errors="coerce")
    ).fillna(np.inf)
    frame = frame.assign(
        _status=status_rank,
        _score=score,
        _alpha_error=alpha_error,
        _frequency_error=frequency_error,
    )
    frame = frame.sort_values(
        ["_status", "_score", "_alpha_error", "_frequency_error"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame.drop(columns=["_status", "_score", "_alpha_error", "_frequency_error"])


def _create_summary_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    successful = frame.loc[frame["status"].eq("success")].copy()
    if successful.empty:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figure_dir = output_dir / "physical_summary" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for column, ylabel, title, filename in (
        (
            "alpha_relative_error_percent",
            "Relativer Fehler [%]",
            "Vergleich der exponentiellen Abklingkonstante",
            "alpha_error_ranking.png",
        ),
        (
            "natural_frequency_relative_error_percent",
            "Relativer Fehler [%]",
            "Vergleich der Eigenfrequenz",
            "natural_frequency_error_ranking.png",
        ),
        (
            "damping_ratio_relative_error_percent",
            "Relativer Fehler [%]",
            "Vergleich des Dämpfungsgrads",
            "damping_ratio_error_ranking.png",
        ),
        (
            "envelope_correlation",
            "Pearson-Korrelation der Hüllkurven",
            "Übereinstimmung der normierten Abklinghüllkurven",
            "envelope_correlation_ranking.png",
        ),
    ):
        values = pd.to_numeric(successful[column], errors="coerce")
        if not values.notna().any():
            continue
        path = figure_dir / filename
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.bar(successful["axis"], values)
        ax.set_xlabel("IMU-Achse")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _overall_assessment(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame.loc[frame["status"].eq("success")].copy()
    if successful.empty:
        return {
            "classification": "physical_validation_failed",
            "message": "Keine Achse konnte physikalisch ausgewertet werden.",
            "recommendations": ["aligned_signals.csv und Ausschwingfenster prüfen."],
        }

    best = successful.iloc[0]
    classification = str(best.get("classification", "unknown"))
    recommendations: list[str] = []
    video_decaying = bool(best.get("video_is_decaying")) if pd.notna(best.get("video_is_decaying")) else False
    imu_decaying = bool(best.get("imu_is_decaying")) if pd.notna(best.get("imu_is_decaying")) else False
    if not video_decaying or not imu_decaying:
        recommendations.append(
            "Beginn und Ende des Ausschwingversuchs explizit mit --decay-start-s und --decay-end-s setzen."
        )
        recommendations.append(
            "Prüfen, ob der ausgewertete Abschnitt noch den Frequenzsweep statt des freien Ausschwingens enthält."
        )
    recommendations.extend(
        [
            "Automatisch erkannte Peaks und Exponentialfits in den Achsenplots visuell kontrollieren.",
            "Dämpfungswerte nur dann berichten, wenn die logarithmierte Hüllkurve näherungsweise linear ist und R² ausreichend hoch ist.",
            "Da Kamera Position und IMU Beschleunigung messen, absolute Amplituden nicht direkt vergleichen; Abklingrate und Frequenz sind die zentralen gemeinsamen Merkmale.",
            "Sensororientierung und zur beobachteten Bewegungsrichtung passende IMU-Achse anhand des Aufbaus begründen.",
            "Die automatische Achsenrangfolge ist eine Auswahlhilfe und kein Signifikanznachweis.",
        ]
    )
    return {
        "classification": classification,
        "message": str(best.get("message", "")),
        "best_axis": str(best["axis"]),
        "best_physical_similarity_score": _finite_float(best.get("physical_similarity_score")),
        "best_alpha_relative_error_percent": _finite_float(best.get("alpha_relative_error_percent")),
        "best_natural_frequency_relative_error_percent": _finite_float(
            best.get("natural_frequency_relative_error_percent")
        ),
        "best_damped_frequency_relative_error_percent": _finite_float(
            best.get("damped_frequency_relative_error_percent")
        ),
        "best_envelope_correlation": _finite_float(best.get("envelope_correlation")),
        "recommendations": recommendations,
        "criteria": [
            "Exponentielle Abklingkonstante alpha",
            "Zeitkonstante tau",
            "Logarithmisches Dekrement",
            "Gedämpfte Eigenfrequenz",
            "Ungedämpfte Eigenfrequenz",
            "Dämpfungsgrad zeta",
            "Korrelation und Fehler der normierten Hüllkurven",
            "Struktur der Hüllkurvenresiduen",
        ],
    }


def _write_html(path: Path, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    display_columns = [
        "rank",
        "axis",
        "classification",
        "physical_similarity_score",
        "alpha_relative_error_percent",
        "natural_frequency_relative_error_percent",
        "damped_frequency_relative_error_percent",
        "damping_ratio_relative_error_percent",
        "envelope_correlation",
        "envelope_rmse",
        "video_alpha_1_per_s",
        "imu_alpha_1_per_s",
        "video_natural_frequency_hz",
        "imu_natural_frequency_hz",
        "video_damping_ratio",
        "imu_damping_ratio",
        "video_exponential_fit_r_squared",
        "imu_exponential_fit_r_squared",
    ]
    available = [column for column in display_columns if column in frame.columns]
    table = frame[available].to_html(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.6g}",
    )
    assessment = summary["assessment"]
    recommendations = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in assessment.get("recommendations", [])
    )
    criteria = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in assessment.get("criteria", [])
    )
    body = f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Physikalische Video–IMU-Validierung</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;overflow-wrap:anywhere}}
th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}
th:first-child,td:first-child{{text-align:center}}
th{{background:#eee}}
.box{{border:1px solid #bbb;padding:1rem;margin:1rem 0}}
.warning{{background:#fff5d6}}
code{{background:#f4f4f4;padding:.1rem .3rem}}
</style>
</head>
<body>
<h1>Physikalische Video–IMU-Validierung des Ausschwingversuchs</h1>
<div class="box">
<strong>{html.escape(str(assessment.get("message", "")))}</strong><br>
Beste Achse: <code>{html.escape(str(assessment.get("best_axis", "n/a")))}</code>
</div>
<div class="box warning">
<strong>Hinweis:</strong> Kamera und IMU messen unterschiedliche physikalische Größen.
Absolute Amplituden und punktweiser RMSE sind daher nicht das Hauptkriterium.
Verglichen werden vor allem Abklingrate, Dämpfung und Eigenfrequenz.
</div>
<h2>Verwendete physikalische Kriterien</h2>
<ul>{criteria}</ul>
<h2>Achsenranking</h2>
{table}
<h2>Empfohlene Kontrollen</h2>
<ul>{recommendations}</ul>
<p><small>Automatisch erzeugt mit validate_video_vs_imu_physical.py.</small></p>
</body>
</html>"""
    path.write_text(body, encoding="utf-8")


def run_physical_validation(
    physical_cfg: PhysicalConfig,
    batch_cfg: Optional[BatchConfig] = None,
    run_base_validation: bool = True,
) -> dict[str, Any]:
    """Führt optional v3 und danach die physikalische Ausschwinganalyse aus."""

    started = perf_counter()
    base_summary: Optional[dict[str, Any]] = None
    if run_base_validation:
        if batch_cfg is None:
            raise ValueError("Für die Basisvalidierung wird eine BatchConfig benötigt.")
        LOGGER.info("1/2 Bestehende v3-Validierung ausführen")
        base_summary = run_axis_comparison(batch_cfg)
    else:
        LOGGER.info("1/2 Vorhandene v3-Ergebnisse wiederverwenden")

    LOGGER.info("2/2 Physikalische Ausschwinganalyse ausführen")
    results: list[AxisPhysicalResult] = []
    for number, axis in enumerate(physical_cfg.axes, start=1):
        LOGGER.info("Physikalische Analyse Achse %d/%d: %s", number, len(physical_cfg.axes), axis)
        aligned_csv = physical_cfg.output_dir / axis / "data" / "aligned_signals.csv"
        results.append(analyze_axis(axis, aligned_csv, physical_cfg))

    comparison = _comparison_frame(results)
    physical_dir = physical_cfg.output_dir / "physical_summary"
    physical_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = physical_dir / "physical_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    plots = _create_summary_plots(comparison, physical_cfg.output_dir)
    assessment = _overall_assessment(comparison)

    summary = {
        "output_dir": str(physical_cfg.output_dir.resolve()),
        "physical_comparison_csv": str(comparison_csv.resolve()),
        "assessment": assessment,
        "configuration": dataclasses.asdict(physical_cfg),
        "base_validation_summary": base_summary,
        "summary_plots": plots,
        "runtime_s": perf_counter() - started,
    }
    summary_json = physical_dir / "physical_summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_html(physical_dir / "physical_summary.html", comparison, summary)
    LOGGER.info("Physikalische Validierung abgeschlossen: %s", comparison_csv)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Physikalische Video–IMU-Validierung für Ausschwingversuche: "
            "Hüllkurve, Abklingkonstante, Dämpfung und Eigenfrequenz."
        ),
    )
    parser.add_argument("--video-csv", "--video-path", dest="video_path", type=Path)
    parser.add_argument("--imu-csv", "--imu-path", dest="imu_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-time-column", default="time_seconds")
    parser.add_argument("--video-value-column", default="inner_pipe_track_center_x")
    parser.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES))
    parser.add_argument("--video-name", default="video")
    parser.add_argument("--imu-name-prefix", default="imu")
    parser.add_argument("--video-unit")
    parser.add_argument("--imu-unit")
    parser.add_argument("--normalize", choices=["none", "zscore", "minmax", "robust"], default="zscore")
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--lowpass-hz", type=float)
    parser.add_argument("--highpass-hz", type=float)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument(
        "--synchronization-method",
        choices=["cross_correlation", "timestamp", "none"],
        default="cross_correlation",
    )
    parser.add_argument("--max-lag-s", type=float)
    parser.add_argument("--manual-lag-s", type=float)
    parser.add_argument("--target-sample-rate-hz", type=float)
    parser.add_argument("--start-time-s", type=float)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Vorhandene <output-dir>/<axis>/data/aligned_signals.csv verwenden und v3 nicht erneut starten.",
    )

    physical = parser.add_argument_group("Physikalische Ausschwinganalyse")
    physical.add_argument(
        "--decay-start-s",
        type=float,
        help="Expliziter Beginn des freien Ausschwingens auf der synchronisierten Zeitachse.",
    )
    physical.add_argument(
        "--decay-end-s",
        type=float,
        help="Explizites Ende des auszuwertenden Ausschwingabschnitts.",
    )
    physical.add_argument(
        "--no-auto-decay-start",
        action="store_true",
        help="Bei fehlendem --decay-start-s direkt am Beginn der synchronisierten Daten starten.",
    )
    physical.add_argument("--auto-start-fraction", type=float, default=0.98)
    physical.add_argument("--minimum-decay-duration-s", type=float, default=5.0)
    physical.add_argument("--envelope-smoothing-s", type=float, default=0.20)
    physical.add_argument(
        "--physical-lowpass-hz",
        type=float,
        help="Optionaler Tiefpass nur für die physikalische Ausschwinganalyse.",
    )
    physical.add_argument("--minimum-peak-distance-s", type=float)
    physical.add_argument("--peak-prominence-fraction", type=float, default=0.05)
    physical.add_argument("--minimum-peaks", type=int, default=6)
    physical.add_argument("--fit-lower-envelope-fraction", type=float, default=0.08)
    physical.add_argument("--fit-upper-envelope-fraction", type=float, default=0.98)
    physical.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        axes = tuple(args.axes)
        physical_cfg = PhysicalConfig(
            output_dir=args.output_dir,
            axes=axes,
            video_name=args.video_name,
            imu_name_prefix=args.imu_name_prefix,
            decay_start_s=args.decay_start_s,
            decay_end_s=args.decay_end_s,
            auto_decay_start=not args.no_auto_decay_start,
            auto_start_fraction=args.auto_start_fraction,
            minimum_decay_duration_s=args.minimum_decay_duration_s,
            envelope_smoothing_s=args.envelope_smoothing_s,
            signal_lowpass_hz=args.physical_lowpass_hz,
            minimum_peak_distance_s=args.minimum_peak_distance_s,
            peak_prominence_fraction=args.peak_prominence_fraction,
            minimum_peaks=args.minimum_peaks,
            fit_lower_envelope_fraction=args.fit_lower_envelope_fraction,
            fit_upper_envelope_fraction=args.fit_upper_envelope_fraction,
            create_plots=not args.no_plots,
        )

        batch_cfg: Optional[BatchConfig] = None
        if not args.reuse_existing:
            if args.video_path is None or args.imu_path is None:
                raise ValueError(
                    "--video-csv und --imu-csv sind erforderlich, sofern "
                    "--reuse-existing nicht verwendet wird."
                )
            batch_cfg = BatchConfig(
                video_path=args.video_path,
                imu_path=args.imu_path,
                output_dir=args.output_dir,
                video_time_column=args.video_time_column,
                video_value_column=args.video_value_column,
                axes=axes,
                video_name=args.video_name,
                imu_name_prefix=args.imu_name_prefix,
                video_unit=args.video_unit,
                imu_unit=args.imu_unit,
                normalize=args.normalize,
                detrend=not args.no_detrend,
                center=not args.no_center,
                smoothing_window=args.smoothing_window,
                lowpass_hz=args.lowpass_hz,
                highpass_hz=args.highpass_hz,
                synchronization_method=args.synchronization_method,
                max_lag_s=args.max_lag_s,
                manual_lag_s=args.manual_lag_s,
                target_sample_rate_hz=args.target_sample_rate_hz,
                start_time_s=args.start_time_s,
                end_time_s=args.end_time_s,
                create_plots=not args.no_plots,
            )

        summary = run_physical_validation(
            physical_cfg,
            batch_cfg=batch_cfg,
            run_base_validation=not args.reuse_existing,
        )
    except KeyboardInterrupt:
        LOGGER.error("Abgebrochen.")
        return 130
    except Exception as exc:
        LOGGER.error("Physikalische Validierung fehlgeschlagen: %s", exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
