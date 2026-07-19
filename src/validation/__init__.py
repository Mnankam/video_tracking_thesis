"""
validation
==========

Video ↔ IMU Validation Package

Dieses Paket stellt alle Module zur Validierung eines Video-Tracking-
Signals gegenüber einer IMU-Referenz bereit.

Module
-------
video_loader
    Laden des Video-Tracking-Signals.

imu_loader
    Laden des IMU-Signals.

preprocessing
    Signalvorverarbeitung.

synchronization
    Zeitliche Synchronisation der Signale.

metrics
    Berechnung der Validierungskennzahlen.

plotting
    Erstellung der Validierungsplots.

report
    Erstellung des Validierungsberichts.
"""

__version__ = "1.0.0"
__author__ = "Serge Kouomnankam"

# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------

from .video_loader import (
    load_video_signal,
)

from .imu_loader import (
    load_imu_signal,
)

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

from .preprocessing import (
    preprocess_signal,
    normalize,
    remove_offset,
    remove_linear_trend,
    resample_signal,
)

# ---------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------

from .synchronization import (
    estimate_time_shift,
    synchronize_signals,
)

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

from .metrics import (
    compute_validation_metrics,
    calculate_pearson_correlation,
    calculate_rmse,
    calculate_normalized_rmse,
    calculate_mae,
    calculate_max_absolute_error,
    calculate_r_squared,
    compute_fft_spectrum,
    find_dominant_frequency,
    find_spectral_peaks,
    calculate_frequency_error,
    save_metrics_csv,
)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

from .plotting import (
    create_validation_plots,
    plot_time_domain_comparison,
    plot_residual_error,
    plot_fft_comparison,
    plot_cross_correlation,
    plot_validation_overview,
)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

from .report import (
    generate_report,
)

# ---------------------------------------------------------------------

__all__ = [
    # Loader
    "load_video_signal",
    "load_imu_signal",

    # Preprocessing
    "preprocess_signal",
    "normalize",
    "remove_offset",
    "remove_linear_trend",
    "resample_signal",

    # Synchronization
    "estimate_time_shift",
    "synchronize_signals",

    # Metrics
    "compute_validation_metrics",
    "calculate_pearson_correlation",
    "calculate_rmse",
    "calculate_normalized_rmse",
    "calculate_mae",
    "calculate_max_absolute_error",
    "calculate_r_squared",
    "compute_fft_spectrum",
    "find_dominant_frequency",
    "find_spectral_peaks",
    "calculate_frequency_error",
    "save_metrics_csv",

    # Plotting
    "create_validation_plots",
    "plot_time_domain_comparison",
    "plot_residual_error",
    "plot_fft_comparison",
    "plot_cross_correlation",
    "plot_validation_overview",

    # Report
    "generate_report",
]