from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "5m_long_four_method_comparison_mpe_0_60"

ORIGINAL_CLOUD = Path("G:/我的云端硬盘/单孔漏斗爆破/5m.TXT")
ORIGINAL_FALLBACK = ROOT / "5m.TXT"
CEEMDAN_CSV = Path(
    "C:/Users/admin/Documents/多种适应vmd/output/"
    "ceemdan_paper_denoising/results/denoised_signal.csv"
)
VMD_SSA_CSV = Path("D:/vmd_k8_output/results/algorithm_SSA_denoised_signal.csv")
VMD_MPE_CSV = (
    ROOT
    / "output"
    / "5m_long_mpe_denoising"
    / "5m_Long_denoised_signal.csv"
)

SERIES = {
    "original": ("Original", "#000000", "-", 0.9),
    "ceemdan": ("CEEMDAN", "#E69F00", "--", 1.0),
    "vmd_ssa": ("VMD-SSA", "#0072B2", "-.", 1.0),
    "vmd_mpe_0_60": ("VMD-MPE (threshold 0.60)", "#009E73", ":", 1.1),
}


def _first_number(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"metadata field does not contain a number: {text!r}")
    return float(match.group(0))


def _read_text_with_fallback(primary: Path, fallback: Path) -> str:
    if primary.exists():
        text = primary.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return text
    if not fallback.exists():
        raise FileNotFoundError(f"missing fallback input file: {fallback}")
    text = fallback.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"fallback input file is empty: {fallback}")
    return text


def load_instantel_long(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return original time, Long velocity, sample rate, and unit."""
    text = _read_text_with_fallback(path, ORIGINAL_FALLBACK)
    lines = text.splitlines()
    metadata = {}
    header_index = None

    for index, line in enumerate(lines):
        stripped = line.strip().strip('"')
        if stripped.split() == ["Tran", "Vert", "Long"]:
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()

    if header_index is None:
        raise ValueError("could not locate Tran Vert Long data header")
    if "Sample Rate" not in metadata:
        raise ValueError("could not locate Sample Rate metadata")
    if "Pre-trigger Length" not in metadata:
        raise ValueError("could not locate Pre-trigger Length metadata")

    fs = _first_number(metadata["Sample Rate"])
    pretrigger = abs(_first_number(metadata["Pre-trigger Length"]))
    unit = metadata.get("Units", "").split()[0] if metadata.get("Units") else "mm/s"

    rows = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 3:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"non-numeric waveform row: {line!r}") from exc

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] == 0:
        raise ValueError("waveform data must contain three finite numeric columns")
    if not np.isfinite(data).all():
        raise ValueError("waveform data contains non-finite values")

    time_s = np.arange(data.shape[0], dtype=float) / fs - pretrigger
    return time_s, data[:, 2], fs, unit


def load_denoised_csv(
    path: Path, original_column: str, denoised_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CSV time, embedded original signal, and denoised signal."""
    if not path.exists():
        raise FileNotFoundError(f"missing denoised CSV: {path}")
    frame = pd.read_csv(path)
    for column in ("time_s", original_column, denoised_column):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing required column {column!r}")
    values = frame[["time_s", original_column, denoised_column]].to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values[:, 0], values[:, 1], values[:, 2]


def align_signals(
    time_s: np.ndarray,
    original: np.ndarray,
    ceemdan: np.ndarray,
    vmd_ssa: np.ndarray,
    vmd_mpe: np.ndarray,
) -> pd.DataFrame:
    arrays = [time_s, original, ceemdan, vmd_ssa, vmd_mpe]
    if len({len(np.asarray(value)) for value in arrays}) != 1:
        raise ValueError("all signals must contain the same number of samples")
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays):
        raise ValueError("time and signals must contain only finite values")
    return pd.DataFrame(
        {
            "time_s": time_s,
            "original": original,
            "ceemdan": ceemdan,
            "vmd_ssa": vmd_ssa,
            "vmd_mpe_0_60": vmd_mpe,
        }
    )


def one_sided_amplitude(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be positive")
    spectrum = np.fft.rfft(values)
    amplitude = np.abs(spectrum) / values.size
    if values.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    frequency = np.fft.rfftfreq(values.size, d=1.0 / fs)
    return frequency, amplitude


def _statistics(frame: pd.DataFrame, fs: float) -> pd.DataFrame:
    rows = []
    for column, (label, _, _, _) in SERIES.items():
        frequency, amplitude = one_sided_amplitude(frame[column].to_numpy(), fs)
        mask = (frequency > 0.0) & (frequency <= 250.0)
        local_peak = int(np.argmax(amplitude[mask]))
        masked_frequency = frequency[mask]
        masked_amplitude = amplitude[mask]
        values = frame[column].to_numpy(dtype=float)
        rows.append(
            {
                "signal": label,
                "peak_abs": float(np.max(np.abs(values))),
                "mean": float(np.mean(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "std": float(np.std(values, ddof=0)),
                "dominant_frequency_0_250_hz": float(masked_frequency[local_peak]),
                "dominant_amplitude_0_250": float(masked_amplitude[local_peak]),
            }
        )
    return pd.DataFrame(rows)


def _fft_frame(frame: pd.DataFrame, fs: float) -> pd.DataFrame:
    fft_columns = {}
    frequency_reference = None
    for column, (label, _, _, _) in SERIES.items():
        frequency, amplitude = one_sided_amplitude(frame[column].to_numpy(), fs)
        mask = frequency <= 250.0
        if frequency_reference is None:
            frequency_reference = frequency[mask]
            fft_columns["frequency_hz"] = frequency_reference
        else:
            np.testing.assert_allclose(frequency[mask], frequency_reference)
        fft_columns[label] = amplitude[mask]
    return pd.DataFrame(fft_columns)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)


def _plot_time(frame: pd.DataFrame, unit: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for column, (label, color, linestyle, linewidth) in SERIES.items():
        ax.plot(
            frame["time_s"],
            frame[column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Velocity ({unit})")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _plot_fft(fft: pd.DataFrame, unit: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for column, (_, color, linestyle, linewidth) in zip(fft.columns[1:], SERIES.values()):
        ax.plot(
            fft["frequency_hz"],
            fft[column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=column,
        )
    ax.set_xlim(0.0, 250.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"One-sided amplitude ({unit})")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _plot_combined(frame: pd.DataFrame, fft: pd.DataFrame, unit: str) -> plt.Figure:
    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(12, 9))
    for column, (label, color, linestyle, linewidth) in SERIES.items():
        ax_time.plot(
            frame["time_s"],
            frame[column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    for column, (_, color, linestyle, linewidth) in zip(fft.columns[1:], SERIES.values()):
        ax_fft.plot(
            fft["frequency_hz"],
            fft[column],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=column,
        )
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel(f"Velocity ({unit})")
    ax_fft.set_xlim(0.0, 250.0)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel(f"One-sided amplitude ({unit})")
    ax_time.legend(frameon=False, ncol=2, loc="upper right")
    ax_fft.legend(frameon=False, ncol=2, loc="upper right")
    _style_axes(ax_time)
    _style_axes(ax_fft)
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_results(frame: pd.DataFrame, fs: float, unit: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "5m_Long_four_signals_aligned.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fft = _fft_frame(frame, fs)
    fft.to_csv(
        output_dir / "5m_Long_fft_0_250Hz.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _statistics(frame, fs).to_csv(
        output_dir / "5m_Long_four_signal_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_figure(_plot_time(frame, unit), output_dir, "5m_Long_time_comparison")
    _save_figure(_plot_fft(fft, unit), output_dir, "5m_Long_fft_0_250Hz")
    _save_figure(
        _plot_combined(frame, fft, unit),
        output_dir,
        "5m_Long_time_fft_comparison",
    )


def load_all_sources() -> tuple[pd.DataFrame, float, str]:
    time_s, original, fs, unit = load_instantel_long(ORIGINAL_CLOUD)
    _, ceemdan_original, ceemdan = load_denoised_csv(
        CEEMDAN_CSV, "original_long", "ceemdan_denoised"
    )
    _, ssa_original, vmd_ssa = load_denoised_csv(
        VMD_SSA_CSV, "original_long_mm_s", "denoised_SSA_mm_s"
    )
    _, mpe_original, vmd_mpe = load_denoised_csv(
        VMD_MPE_CSV, "original", "denoised"
    )
    for label, embedded in {
        "CEEMDAN": ceemdan_original,
        "VMD-SSA": ssa_original,
        "VMD-MPE": mpe_original,
    }.items():
        np.testing.assert_allclose(
            embedded,
            original,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{label} embedded original does not match 5m Long",
        )
    return align_signals(time_s, original, ceemdan, vmd_ssa, vmd_mpe), fs, unit


def main() -> None:
    frame, fs, unit = load_all_sources()
    export_results(frame, fs, unit, OUTPUT_DIR)
    print(f"aligned_samples={len(frame)}")
    print(f"sample_rate_hz={fs:g}")
    print(f"output_dir={OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
