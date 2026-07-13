from __future__ import annotations
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_signal(df, tc, sc):
    missing = [c for c in (tc, sc) if c not in df.columns]
    if missing:
        raise KeyError("Fehlende Spalten: " + ", ".join(missing))
    d = df[[tc, sc]].copy()
    d[tc] = pd.to_numeric(d[tc], errors="coerce")
    d[sc] = pd.to_numeric(d[sc], errors="coerce")
    d = d.dropna().sort_values(tc).groupby(tc, as_index=False)[sc].mean()
    if len(d) < 8:
        raise ValueError("Zu wenige gültige Messpunkte.")
    return d[tc].to_numpy(float), d[sc].to_numpy(float)


def sampling_rate(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Abtastzeit nicht bestimmbar.")
    return float(1.0 / np.median(dt))


def detrend(t, x):
    p = np.polyfit(t, x, 1)
    return x - np.polyval(p, t)


def spectrum(t, x):
    fs = sampling_rate(t)
    y = detrend(t, x)
    w = np.hanning(len(y))
    Y = np.fft.rfft(y * w)
    f = np.fft.rfftfreq(len(y), d=1.0 / fs)
    cg = np.sum(w) / len(w)
    a = np.abs(Y) / (len(y) * cg)
    if len(a) > 2:
        a[1:-1] *= 2.0
    return f, a


def dominant_frequency(t, x, fmin, fmax):
    f, a = spectrum(t, x)
    valid = (f >= fmin) & (f > 0)
    if fmax is not None:
        valid &= f <= fmax
    if not np.any(valid):
        raise ValueError("Kein gültiger FFT-Frequenzbereich.")
    ids = np.where(valid)[0]
    i = ids[np.argmax(a[valid])]
    return float(f[i]), f, a


def robust_amplitude(x):
    return float((np.percentile(x, 95) - np.percentile(x, 5)) / 2.0)


def relerr(ref, value):
    if np.isclose(ref, 0):
        return np.nan
    return float(abs(value - ref) / abs(ref) * 100.0)


def lowpass_fir(fs, target_fs, taps=101, cutoff_ratio=0.90):
    if taps % 2 == 0:
        taps += 1
    cutoff = cutoff_ratio * target_fs / 2.0
    n = np.arange(taps) - (taps - 1) / 2.0
    fc = cutoff / fs
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.hamming(taps)
    h /= np.sum(h)
    return h


def zero_phase_fir(x, h):
    pad = min(len(x) - 1, 3 * (len(h) - 1))
    if pad < 1:
        return np.convolve(x, h, mode="same")
    xp = np.pad(x, pad, mode="reflect")
    y = np.convolve(xp, h, mode="same")
    return y[pad:-pad]


def downsample(t, x, original_fs, target_fs):
    tol = max(1e-6, abs(original_fs) * 1e-6)
    if target_fs <= 0:
        raise ValueError("Zielabtastrate muss > 0 Hz sein.")
    if target_fs > original_fs + tol:
        raise ValueError("Zielabtastrate ist größer als Originalabtastrate.")
    step = max(1, int(round(original_fs / target_fs)))
    actual_fs = original_fs / step
    if step == 1:
        return t.copy(), x.copy(), float(actual_fs), 0
    h = lowpass_fir(original_fs, actual_fs)
    xf = zero_phase_fir(x, h)
    return t[::step], xf[::step], float(actual_fs), 1


def peak_near(f, a, target, width, fmax):
    lo = max(0.0, target - width)
    hi = min(fmax, target + width)
    valid = (f >= lo) & (f <= hi)
    if not np.any(valid):
        return np.nan, np.nan
    ids = np.where(valid)[0]
    i = ids[np.argmax(a[valid])]
    return float(f[i]), float(a[i])


def harmonic_table(t, x, f0, nh, width, fmax):
    f, a = spectrum(t, x)
    nyq = sampling_rate(t) / 2.0
    limit = min(nyq, fmax)
    rows = []
    for k in range(1, nh + 1):
        expected = k * f0
        observable = expected < nyq and expected <= limit
        if observable:
            fm, am = peak_near(f, a, expected, width, limit)
        else:
            fm, am = np.nan, np.nan
        rows.append({
            "harmonic_order": k,
            "expected_frequency_hz": expected,
            "measured_frequency_hz": fm,
            "spectral_amplitude_px": am,
            "observable_below_nyquist": int(observable),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="Inner-pipe validation with anti-aliasing and harmonic analysis."
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--time-column", default="time_seconds")
    p.add_argument("--signal-column", default="inner_pipe_track_center_x")
    p.add_argument("--target-fps", nargs="+", type=float,
                   default=[200, 100, 50, 25, 20, 10])
    p.add_argument("--min-frequency", type=float, default=0.5)
    p.add_argument("--max-frequency", type=float, default=50.0)
    p.add_argument("--harmonics", type=int, default=6)
    p.add_argument("--harmonic-search-width", type=float, default=0.25)
    p.add_argument("--acceptance-threshold-percent", type=float, default=10.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    t, x = prepare_signal(df, args.time_column, args.signal_column)

    fs = sampling_rate(t)
    duration = float(t[-1] - t[0])
    nyq = fs / 2.0
    fmax = min(args.max_frequency, nyq)
    f0, f_ref, a_ref = dominant_frequency(t, x, args.min_frequency, fmax)
    amp_ref = robust_amplitude(x)

    href = harmonic_table(
        t, x, f0, args.harmonics, args.harmonic_search_width, fmax
    )
    href.to_csv(os.path.join(args.out_dir, "harmonic_reference.csv"), index=False)
    ref = href.set_index("harmonic_order").to_dict("index")

    ds_rows, sp_rows, spectra = [], [], {}

    for requested in args.target_fps:
        if requested > fs + max(1e-6, fs * 1e-6):
            print(f"Übersprungen: {requested:.6f} Hz > {fs:.6f} Hz")
            continue

        td, xd, fsd, aa = downsample(t, x, fs, requested)
        if len(xd) < 8:
            continue

        nd = fsd / 2.0
        fdmax = min(args.max_frequency, nd)
        fd, ff, aa_spec = dominant_frequency(
            td, xd, args.min_frequency, fdmax
        )
        amp = robust_amplitude(xd)
        ef = relerr(f0, fd)
        ea = relerr(amp_ref, amp)

        ds_rows.append({
            "requested_sampling_rate_hz": requested,
            "actual_sampling_rate_hz": fsd,
            "nyquist_frequency_hz": nd,
            "number_of_samples": len(xd),
            "duration_s": float(td[-1] - td[0]),
            "dominant_frequency_hz": fd,
            "frequency_relative_error_percent": ef,
            "amplitude_px": amp,
            "amplitude_relative_error_percent": ea,
            "nyquist_condition_satisfied_for_fundamental": int(f0 < nd),
            "frequency_within_acceptance_threshold": int(
                ef <= args.acceptance_threshold_percent
            ),
            "amplitude_within_acceptance_threshold": int(
                ea <= args.acceptance_threshold_percent
            ),
            "anti_aliasing_applied": aa,
        })

        spectra[fsd] = (ff, aa_spec)
        hd = harmonic_table(
            td, xd, f0, args.harmonics,
            args.harmonic_search_width, fdmax
        )

        for _, r in hd.iterrows():
            k = int(r["harmonic_order"])
            rr = ref[k]
            observable = int(r["observable_below_nyquist"])
            efh = relerr(rr["measured_frequency_hz"], r["measured_frequency_hz"])                 if observable and np.isfinite(r["measured_frequency_hz"]) else np.nan
            eah = relerr(rr["spectral_amplitude_px"], r["spectral_amplitude_px"])                 if observable and np.isfinite(r["spectral_amplitude_px"]) else np.nan
            preserved = int(
                observable and np.isfinite(efh) and np.isfinite(eah)
                and efh <= args.acceptance_threshold_percent
                and eah <= args.acceptance_threshold_percent
            )
            sp_rows.append({
                "requested_sampling_rate_hz": requested,
                "actual_sampling_rate_hz": fsd,
                "nyquist_frequency_hz": nd,
                "harmonic_order": k,
                "expected_frequency_hz": r["expected_frequency_hz"],
                "reference_frequency_hz": rr["measured_frequency_hz"],
                "measured_frequency_hz": r["measured_frequency_hz"],
                "frequency_relative_error_percent": efh,
                "reference_spectral_amplitude_px": rr["spectral_amplitude_px"],
                "measured_spectral_amplitude_px": r["spectral_amplitude_px"],
                "spectral_amplitude_relative_error_percent": eah,
                "observable_below_nyquist": observable,
                "preserved_within_acceptance_threshold": preserved,
            })

    ds = pd.DataFrame(ds_rows)
    sp = pd.DataFrame(sp_rows)
    ds.to_csv(os.path.join(args.out_dir, "downsampling_validation.csv"), index=False)
    sp.to_csv(os.path.join(args.out_dir, "spectral_preservation.csv"), index=False)

    summary = pd.DataFrame([{
        "input_csv": args.csv,
        "signal_column": args.signal_column,
        "number_of_samples": len(x),
        "duration_s": duration,
        "original_sampling_rate_hz": fs,
        "original_nyquist_frequency_hz": nyq,
        "frequency_resolution_hz": 1.0 / duration,
        "original_dominant_frequency_hz": f0,
        "original_amplitude_px": amp_ref,
        "signal_min_px": float(np.min(x)),
        "signal_max_px": float(np.max(x)),
        "signal_mean_px": float(np.mean(x)),
        "signal_std_px": float(np.std(x, ddof=1)),
        "number_of_harmonics_evaluated": args.harmonics,
        "acceptance_threshold_percent": args.acceptance_threshold_percent,
        "anti_aliasing_downsampling": 1,
    }])
    summary.to_csv(os.path.join(args.out_dir, "validation_summary.csv"), index=False)

    ps = sp.groupby(
        ["requested_sampling_rate_hz", "actual_sampling_rate_hz",
         "nyquist_frequency_hz"], as_index=False
    ).agg(
        number_of_reference_harmonics=("harmonic_order", "count"),
        number_observable_below_nyquist=("observable_below_nyquist", "sum"),
        number_preserved_within_threshold=(
            "preserved_within_acceptance_threshold", "sum"
        ),
    )
    ps["observable_fraction_percent"] = (
        100 * ps["number_observable_below_nyquist"]
        / ps["number_of_reference_harmonics"]
    )
    ps["preserved_fraction_percent"] = (
        100 * ps["number_preserved_within_threshold"]
        / ps["number_of_reference_harmonics"]
    )
    ps.to_csv(
        os.path.join(args.out_dir, "spectral_preservation_summary.csv"),
        index=False
    )

    plt.figure(figsize=(12, 4))
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Inner pipe position x [pixel]")
    plt.title("GX010044 - Inner pipe tracking signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "inner_pipe_timeseries.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(f_ref, a_ref)
    for k in range(1, args.harmonics + 1):
        if k * f0 <= fmax:
            plt.axvline(k * f0, linestyle="--" if k == 1 else ":")
    plt.xlim(0, fmax)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Single-sided amplitude [pixel]")
    plt.title("Frequency spectrum and expected harmonics")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "harmonic_spectrum.png"), dpi=300)
    plt.close()

    sd = ds.sort_values("actual_sampling_rate_hz")
    plt.figure(figsize=(10, 4))
    plt.plot(sd["actual_sampling_rate_hz"],
             sd["frequency_relative_error_percent"], marker="o",
             label="Dominant frequency error")
    plt.plot(sd["actual_sampling_rate_hz"],
             sd["amplitude_relative_error_percent"], marker="o",
             label="Robust amplitude error")
    plt.axhline(args.acceptance_threshold_percent, linestyle="--",
                label=f"Acceptance threshold = {args.acceptance_threshold_percent:.1f}%")
    plt.xlabel("Sampling rate [Hz]")
    plt.ylabel("Relative error [%]")
    plt.title("Validation errors after anti-aliased downsampling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "downsampling_error_comparison.png"),
                dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    for fsd in sorted(spectra, reverse=True):
        ff, aa = spectra[fsd]
        valid = ff <= min(args.max_frequency, fsd / 2.0)
        plt.plot(ff[valid], aa[valid], label=f"{fsd:.0f} Hz")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Single-sided amplitude [pixel]")
    plt.title("Spectral preservation after anti-aliased downsampling")
    plt.grid(True)
    plt.legend(title="Sampling rate")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "downsampling_spectrum_comparison.png"),
                dpi=300)
    plt.close()

    print("Validation completed.")
    print(f"Input samples: {len(x)}")
    print(f"Duration: {duration:.6f} s")
    print(f"Sampling rate: {fs:.6f} Hz")
    print(f"Frequency resolution: {1.0 / duration:.6f} Hz")
    print(f"Dominant frequency: {f0:.6f} Hz")
    print(f"Robust amplitude: {amp_ref:.6f} px")
    print("Anti-aliasing: enabled")
    print("Created: validation_summary.csv")
    print("Created: downsampling_validation.csv")
    print("Created: harmonic_reference.csv")
    print("Created: spectral_preservation.csv")
    print("Created: spectral_preservation_summary.csv")


if __name__ == "__main__":
    main()