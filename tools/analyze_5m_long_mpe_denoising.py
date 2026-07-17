from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "vmd_all_modes.xlsx"
OUTPUT_DIR = ROOT / "output" / "5m_long_mpe_denoising"

SHEET_NAME = "5m_Long"
EMBEDDING_DIMENSION = 6
DELAY = 1
MAX_SCALE = 5
MPE_THRESHOLD = 0.6


def format_threshold(value: float) -> str:
    """Display thresholds without rounding 0.55 to 0.6."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def normalized_permutation_entropy(
    signal: np.ndarray,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    delay: int = DELAY,
) -> float:
    """Bandt-Pompe permutation entropy normalized by ln(m!)."""
    values = np.asarray(signal, dtype=float)
    span = (embedding_dimension - 1) * delay + 1
    if values.ndim != 1 or values.size < span:
        raise ValueError("signal is too short for permutation entropy")
    if not np.isfinite(values).all():
        raise ValueError("signal contains non-finite values")

    windows = sliding_window_view(values, span)[:, ::delay]
    # Stable sorting gives deterministic ordering if equal values occur.
    ordinal_patterns = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(ordinal_patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(math.factorial(embedding_dimension)))


def multiscale_permutation_entropy(
    signal: np.ndarray,
    max_scale: int = MAX_SCALE,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    delay: int = DELAY,
) -> tuple[np.ndarray, float]:
    """Return PE at scales 1..max_scale and their arithmetic mean."""
    values = np.asarray(signal, dtype=float)
    scale_values = []
    for scale in range(1, max_scale + 1):
        usable = (values.size // scale) * scale
        coarse_grained = values[:usable].reshape(-1, scale).mean(axis=1)
        scale_values.append(
            normalized_permutation_entropy(
                coarse_grained,
                embedding_dimension=embedding_dimension,
                delay=delay,
            )
        )
    scale_values = np.asarray(scale_values, dtype=float)
    return scale_values, float(np.mean(scale_values))


def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Global Hann-window spectral peak, excluding DC."""
    values = np.asarray(signal, dtype=float)
    windowed = (values - np.mean(values)) * np.hanning(values.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs)
    peak_index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[peak_index])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"missing input workbook: {INPUT_XLSX}")
    signal_frame = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    summary = pd.read_excel(INPUT_XLSX, sheet_name="summary")
    summary = summary[
        (summary["distance"] == "5m")
        & (summary["direction"] == "Long")
    ].sort_values("mode")

    mode_columns = [
        column for column in signal_frame.columns if column.startswith("mode_")
    ]
    if not mode_columns:
        raise ValueError(f"{SHEET_NAME} contains no mode columns")
    if len(summary) != len(mode_columns):
        raise ValueError("summary row count does not match mode count")
    if signal_frame[["time_s", "original", *mode_columns]].isna().any().any():
        raise ValueError(f"{SHEET_NAME} contains missing values")
    return signal_frame, summary


def analyze_modes(
    signal_frame: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mode_columns = [
        column for column in signal_frame.columns if column.startswith("mode_")
    ]
    modes = signal_frame[mode_columns].to_numpy(dtype=float).T
    total_mode_energy = float(np.sum(modes**2))
    rows = []

    for index, (column, mode) in enumerate(zip(mode_columns, modes)):
        scale_values, mean_mpe = multiscale_permutation_entropy(mode)
        is_noise = bool(mean_mpe > MPE_THRESHOLD)
        rows.append(
            {
                "mode": index + 1,
                "column": column,
                "initial_center_hz": float(
                    summary.iloc[index]["initial_center_hz"]
                ),
                "final_center_hz": float(
                    summary.iloc[index]["final_center_hz"]
                ),
                **{
                    f"pe_scale_{scale}": float(scale_values[scale - 1])
                    for scale in range(1, MAX_SCALE + 1)
                },
                "mean_mpe": mean_mpe,
                "mpe_threshold": MPE_THRESHOLD,
                "is_noise": is_noise,
                "decision": "noise" if is_noise else "retained",
                "mode_energy": float(np.sum(mode**2)),
                "mode_energy_percent": float(
                    100.0 * np.sum(mode**2) / total_mode_energy
                ),
                "mode_rms": float(np.sqrt(np.mean(mode**2))),
                "mode_peak_abs": float(np.max(np.abs(mode))),
            }
        )

    mode_metrics = pd.DataFrame(rows)
    keep_mask = ~mode_metrics["is_noise"].to_numpy(dtype=bool)
    denoised = np.sum(modes[keep_mask], axis=0)
    removed_noise = np.sum(modes[~keep_mask], axis=0)
    return mode_metrics, modes, denoised, removed_noise


def evaluate_denoising(
    original: np.ndarray,
    denoised: np.ndarray,
    removed_noise: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    difference = original - denoised
    signal_energy = float(np.sum(original**2))
    difference_energy = float(np.sum(difference**2))
    snr_db = float(10.0 * np.log10(signal_energy / difference_energy))
    rmse = float(np.sqrt(np.mean(difference**2)))
    original_peak = float(np.max(np.abs(original)))
    denoised_peak = float(np.max(np.abs(denoised)))
    original_energy = float(np.sum(original**2))
    denoised_energy = float(np.sum(denoised**2))

    rows = [
        ("SNR", snr_db, "dB", "larger is better (paper Eq. 19)"),
        ("RMSE", rmse, "mm/s", "smaller is better (paper Eq. 20)"),
        ("Original peak absolute velocity", original_peak, "mm/s", "reference"),
        ("Denoised peak absolute velocity", denoised_peak, "mm/s", "small change preferred"),
        (
            "Peak change",
            100.0 * (denoised_peak / original_peak - 1.0),
            "%",
            "absolute change near zero preferred",
        ),
        ("Original squared-amplitude energy", original_energy, "(mm/s)^2", "reference"),
        ("Denoised squared-amplitude energy", denoised_energy, "(mm/s)^2", "high retention preferred"),
        (
            "Energy retained",
            100.0 * denoised_energy / original_energy,
            "%",
            "high retention preferred",
        ),
        (
            "Removed component energy",
            float(np.sum(removed_noise**2)),
            "(mm/s)^2",
            "reported for transparency",
        ),
        (
            "Correlation original-denoised",
            float(np.corrcoef(original, denoised)[0, 1]),
            "-",
            "closer to 1 indicates waveform preservation",
        ),
        (
            "Original global dominant frequency",
            dominant_frequency(original, fs),
            "Hz",
            "supplementary FFT metric, not paper AOK",
        ),
        (
            "Denoised global dominant frequency",
            dominant_frequency(denoised, fs),
            "Hz",
            "supplementary FFT metric, not paper AOK",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "interpretation"])


def plot_modes(
    time_s: np.ndarray,
    modes: np.ndarray,
    mode_metrics: pd.DataFrame,
    unit: str,
) -> None:
    colors = {False: "#0072B2", True: "#D55E00"}
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    for index, (axis, mode) in enumerate(zip(axes.flat, modes)):
        row = mode_metrics.iloc[index]
        is_noise = bool(row["is_noise"])
        axis.plot(time_s, mode, color=colors[is_noise], linewidth=0.65)
        axis.set_ylabel(f"Mode {index + 1}\n({unit})")
        axis.text(
            0.985,
            0.88,
            f"Mean MPE = {row['mean_mpe']:.4f}\n"
            f"{row['decision'].upper()} (threshold {format_threshold(MPE_THRESHOLD)})",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=colors[is_noise],
            bbox={"facecolor": "white", "edgecolor": colors[is_noise], "alpha": 0.9},
        )
        axis.set_title(
            f"Final center = {row['final_center_hz']:.2f} Hz; "
            f"energy = {row['mode_energy_percent']:.3f}%",
            fontsize=9,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15, linewidth=0.4)

    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        "5m/Long VMD modes classified by mean multiscale permutation entropy",
        fontsize=13,
    )
    figure.savefig(OUTPUT_DIR / "5m_Long_modes_MPE.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "5m_Long_modes_MPE.pdf")
    plt.close(figure)


def plot_before_after(
    time_s: np.ndarray,
    original: np.ndarray,
    denoised: np.ndarray,
    removed_noise: np.ndarray,
    unit: str,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
    )
    plots = [
        (original, "Original 5m/Long", "#202020"),
        (denoised, "Denoised (retained modes)", "#0072B2"),
        (removed_noise, "Removed noise modes", "#D55E00"),
    ]
    for axis, (values, title, color) in zip(axes, plots):
        axis.plot(time_s, values, color=color, linewidth=0.65)
        axis.set_ylabel(f"Velocity\n({unit})")
        axis.set_title(title, loc="left", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15, linewidth=0.4)
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(OUTPUT_DIR / "5m_Long_before_after_denoising.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "5m_Long_before_after_denoising.pdf")
    plt.close(figure)


def write_report(
    mode_metrics: pd.DataFrame,
    evaluation: pd.DataFrame,
    noise_modes: list[int],
    signal_frame: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        formatted = frame.copy()
        for column in formatted.select_dtypes(include=[np.number]).columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.6g}")
        headers = [str(column) for column in formatted.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in formatted.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    metric_values = evaluation.set_index("metric")["value"]
    fs = float(summary.iloc[0]["fs_hz"])
    unit = str(summary.iloc[0]["unit"])
    report = f"""# 5m/Long 多尺度排列熵噪声模态分析

## 文件与数据概况

- 输入文件：`{INPUT_XLSX.name}`（Excel OOXML表格）。
- 工作表：`{SHEET_NAME}`。
- 样本数：{len(signal_frame)}；列数：{len(signal_frame.columns)}；缺失值：{int(signal_frame.isna().sum().sum())}。
- 采样频率：{fs:g} Hz；时间范围：{signal_frame['time_s'].iloc[0]:.6g} 至 {signal_frame['time_s'].iloc[-1]:.6g} s。
- 振速单位：{unit}；VMD模态数：{len(mode_metrics)}。

## 方法

- 依据：*Applied Sciences* 2023, 13, 3322，第3.2、3.3和4.1节。
- MPE参数：嵌入维数 m={EMBEDDING_DIMENSION}，延迟 tau={DELAY}，尺度1-{MAX_SCALE}。
- 模态MPE：尺度1-{MAX_SCALE}归一化排列熵的算术平均。
- 噪声判据：平均MPE > {MPE_THRESHOLD}。
- 判为噪声并删除的模态：{noise_modes}。
- 去噪信号：所有非噪声模态求和。
- 效果指标：SNR与RMSE分别按论文公式(19)、(20)计算。

## 各模态计算结果

{markdown_table(mode_metrics)}

## 去噪效果数据

{markdown_table(evaluation)}

## 主要结论

- 仅Mode 8的平均MPE={mode_metrics.loc[mode_metrics['mode'] == 8, 'mean_mpe'].iloc[0]:.6f}超过0.6，故按论文阈值将其判为噪声模态。
- 删除Mode 8后，SNR={metric_values['SNR']:.3f} dB，RMSE={metric_values['RMSE']:.6f} {unit}。
- 峰值变化={metric_values['Peak change']:.3f}%，能量保留={metric_values['Energy retained']:.3f}%，原始与去噪波形相关系数={metric_values['Correlation original-denoised']:.6f}。
- 按论文“较大SNR、较小RMSE、峰值和主频变化较小”的方向性标准，结果显示波形保真度较高；但没有干净真值，不能把这些指标解释为真实噪声误差。

## 建议与限制

- 论文对实测信号的SNR和RMSE本质上比较去噪结果与含噪原始测量值，主要反映保真度，而非真实去噪误差。
- 本报告的全局主频采用Hann窗FFT，只作为补充检查；论文主频保持性采用AOK时频分析，二者不可完全等同。
- Mode 8最终中心频率约266.83 Hz，略高于论文现场案例中主要能量0-250 Hz的范围；这一事实支持噪声判断，但正式判别仍以平均MPE阈值0.6为准。
- 建议在其他炮次或通道上复核阈值稳定性，并保存人工检查结果，避免把短时高频有效冲击误删。
"""
    (OUTPUT_DIR / "5m_Long_MPE_denoising_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_frame, summary = load_inputs()
    mode_metrics, modes, denoised, removed_noise = analyze_modes(
        signal_frame, summary
    )

    time_s = signal_frame["time_s"].to_numpy(dtype=float)
    original = signal_frame["original"].to_numpy(dtype=float)
    fs = float(summary.iloc[0]["fs_hz"])
    unit = str(summary.iloc[0]["unit"])
    evaluation = evaluate_denoising(original, denoised, removed_noise, fs)
    noise_modes = mode_metrics.loc[mode_metrics["is_noise"], "mode"].tolist()

    signal_output = pd.DataFrame(
        {
            "time_s": time_s,
            "original": original,
            "denoised": denoised,
            "removed_noise": removed_noise,
        }
    )
    mode_metrics.to_csv(
        OUTPUT_DIR / "5m_Long_mode_MPE_metrics.csv", index=False, encoding="utf-8-sig"
    )
    evaluation.to_csv(
        OUTPUT_DIR / "5m_Long_denoising_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    signal_output.to_csv(
        OUTPUT_DIR / "5m_Long_denoised_signal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        OUTPUT_DIR / "5m_Long_MPE_denoising_results.xlsx", engine="openpyxl"
    ) as writer:
        mode_metrics.to_excel(writer, sheet_name="mode_metrics", index=False)
        evaluation.to_excel(writer, sheet_name="evaluation", index=False)
        signal_output.to_excel(writer, sheet_name="denoised_signal", index=False)

    plot_modes(time_s, modes, mode_metrics, unit)
    plot_before_after(time_s, original, denoised, removed_noise, unit)
    write_report(
        mode_metrics,
        evaluation,
        noise_modes,
        signal_frame,
        summary,
    )

    print(mode_metrics.to_string(index=False))
    print()
    print(evaluation.to_string(index=False))
    print()
    print(f"Noise modes: {noise_modes}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
