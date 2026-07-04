from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from build_blast_multichannel_stvmd_notebook import (
    DIAGNOSTICS,
    STVMD as MULTICHANNEL_BATCHED_STVMD,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "single_waveform_stvmd_batched.ipynb"


def markdown(source, cell_id):
    return new_markdown_cell(source, id=cell_id)


def code(source, cell_id, tags=()):
    return new_code_cell(
        source,
        id=cell_id,
        metadata={"tags": list(tags)} if tags else {},
    )


def single_stvmd_source():
    source = MULTICHANNEL_BATCHED_STVMD
    source = source.replace(
        "REPOSITORY_WINDOWS = (8, 16, 32, 64, 128, 256)\n\n\n",
        "",
    )
    start = source.index("def validate_config(")
    end = source.index("\n\ndef _pad_width", start)
    validation = r'''def validate_config(
    K, alpha, window_length, batch_windows, max_iters, tau, tol
):
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K 必须为不小于2的整数")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ALPHA 必须为有限正数")
    if not isinstance(window_length, (int, np.integer)) or window_length < 2:
        raise ValueError("WINDOW_LENGTH 必须为不小于2的整数")
    if not isinstance(batch_windows, (int, np.integer)) or batch_windows < 1:
        raise ValueError("BATCH_WINDOWS 必须为正整数")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS 必须为不小于2的整数")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("TAU 必须为有限正数")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("TOL 必须为有限正数")'''
    source = source[:start] + validation + source[end:]
    old_call = (
        "validate_config(K, alpha, window_length, batch_windows, max_iters)"
    )
    new_call = (
        "validate_config(\n"
        "        K, alpha, window_length, batch_windows, max_iters, tau, tol\n"
        "    )"
    )
    if source.count(old_call) != 1:
        raise RuntimeError("Unexpected shared STVMD validation call")
    return source.replace(old_call, new_call)


IMPORTS = r'''
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython.display import display
'''.strip()


CONFIG = r'''
# 只在这里修改输入文件、方向和STVMD参数。
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
WINDOW_LENGTH = 64
TAU = 1e-5
TOL = 1e-5
MAX_ITERS = 300
BATCH_WINDOWS = 128

PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True

print(
    f"file={INPUT_FILE}, direction={DIRECTION}, K={K}, "
    f"alpha={ALPHA}, window={WINDOW_LENGTH}, "
    f"batch={BATCH_WINDOWS}, max_iters={MAX_ITERS}"
)
'''.strip()


LOADER = r'''
@dataclass(frozen=True)
class InstantelRecord:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    data: np.ndarray


@dataclass(frozen=True)
class SingleWaveform:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    direction: str
    time_s: np.ndarray
    values: np.ndarray


def _metadata_number(metadata, key):
    import re

    if key not in metadata:
        raise ValueError(f"缺少元数据字段: {key}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", metadata[key])
    if match is None:
        raise ValueError(f"无法解析元数据字段 {key}: {metadata[key]!r}")
    return float(match.group())


def load_instantel_txt(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    metadata = {}
    header_index = None
    for index, raw in enumerate(lines):
        stripped = raw.strip().strip('"')
        if all(name in stripped for name in ("Tran", "Vert", "Long")):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()
    if header_index is None:
        raise ValueError(f"{path.name}: 未找到 Tran/Vert/Long 数据表头")
    data = np.loadtxt(lines[header_index + 1 :], dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] != 3:
        raise ValueError(
            f"{path.name}: 期望3列数据，实际为{data.shape[1]}列"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: 数据包含 NaN 或无穷值")
    fs = _metadata_number(metadata, "Sample Rate")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    if fs <= 0:
        raise ValueError(f"{path.name}: 采样率必须为正数")
    return InstantelRecord(
        path=path,
        metadata=metadata,
        fs=fs,
        pretrigger_seconds=pretrigger,
        data=data,
    )


def load_single_waveform(path, direction):
    direction_names = ("Tran", "Vert", "Long")
    if direction not in direction_names:
        raise ValueError(f"DIRECTION 必须为 {direction_names} 之一")
    record = load_instantel_txt(path)
    column = direction_names.index(direction)
    values = record.data[:, column].astype(float, copy=False)
    time_s = np.arange(values.size) / record.fs - record.pretrigger_seconds
    return SingleWaveform(
        path=record.path,
        metadata=record.metadata,
        fs=record.fs,
        pretrigger_seconds=record.pretrigger_seconds,
        direction=direction,
        time_s=time_s,
        values=values,
    )
'''.strip()


PLOTTING = r'''
def _mode_color(mode):
    return plt.get_cmap("tab10")(mode % 10)


def _frequency_extent(waveform, frequency_bins):
    return [
        waveform.time_s[0],
        waveform.time_s[-1],
        0.0,
        waveform.fs / 2.0,
    ]


def _limit_frequency_axis(axis, plot_max_hz, fs):
    axis.set_ylim(0.0, min(float(plot_max_hz), fs / 2.0))
    axis.set_ylabel("Frequency (Hz)")


def plot_input_and_tf(waveform, result, plot_max_hz):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.2), constrained_layout=True
    )
    axes[0].plot(waveform.time_s, waveform.values, lw=0.8)
    axes[0].axvline(0, color="black", ls="--", lw=0.8)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Velocity (mm/s)",
        title=f"{waveform.path.name}: {waveform.direction}",
    )
    image = axes[1].imshow(
        power_to_db(result["mean_tf_power"]),
        origin="lower",
        aspect="auto",
        extent=_frequency_extent(
            waveform, result["mean_tf_power"].shape[0]
        ),
        cmap="viridis",
        vmin=-80,
        vmax=0,
    )
    _limit_frequency_axis(axes[1], plot_max_hz, waveform.fs)
    axes[1].set(xlabel="Time (s)", title="Time-frequency power")
    fig.colorbar(image, ax=axes[1], label="Relative power (dB)")
    return fig


def plot_modes(waveform, result):
    modes = result["modes"][:, 0, :]
    fig, axes = plt.subplots(
        modes.shape[0],
        1,
        figsize=(11, max(4.0, 2.0 * modes.shape[0])),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for mode in range(modes.shape[0]):
        axis = axes[mode, 0]
        label = "Residual" if mode == 0 else f"Mode {mode}"
        axis.plot(
            waveform.time_s,
            modes[mode],
            color=_mode_color(mode),
            lw=0.75,
        )
        axis.axvline(0, color="black", ls="--", lw=0.6)
        axis.set_ylabel(f"{label}\nVelocity (mm/s)")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(
        f"{waveform.path.name} {waveform.direction}: dynamic STVMD modes"
    )
    return fig


def plot_if_and_reconstruction(waveform, result, plot_max_hz):
    mode_n = result["modes"].shape[0]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": (1.2, 1.2, 0.8)},
        constrained_layout=True,
    )
    for mode in range(1, mode_n):
        axes[0].plot(
            waveform.time_s,
            result["center_freq_hz"][mode],
            color=_mode_color(mode),
            lw=1.0,
            label=f"Mode {mode}",
        )
    axes[0].axvline(0, color="black", ls="--", lw=0.7)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title="Instantaneous center frequencies",
        ylim=(0.0, min(float(plot_max_hz), waveform.fs / 2.0)),
    )
    axes[0].legend(frameon=False, ncol=max(1, mode_n - 1))

    axes[1].plot(
        waveform.time_s,
        waveform.values,
        color="#64748b",
        lw=0.8,
        label="Input",
    )
    axes[1].plot(
        waveform.time_s,
        result["reconstruction"][0],
        color="#D55E00",
        lw=0.7,
        label="Reconstruction",
    )
    axes[1].axvline(0, color="black", ls="--", lw=0.6)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Velocity (mm/s)",
        title=f"Reconstruction  NRMSE={result['nrmse'][0]:.3g}",
    )
    axes[1].legend(frameon=False)

    fractions = result["energy_fraction"][:, 0]
    bars = axes[2].bar(
        np.arange(mode_n),
        fractions,
        color=[_mode_color(mode) for mode in range(mode_n)],
    )
    axes[2].set(
        xlabel="Component",
        ylabel="Energy fraction",
        title="Mode energy fraction",
    )
    axes[2].set_xticks(
        np.arange(mode_n),
        ["Residual"] + [f"Mode {mode}" for mode in range(1, mode_n)],
    )
    axes[2].bar_label(bars, fmt="%.3f", fontsize=8)
    return fig


def plot_spectrum_if_mapping(waveform, result, plot_max_hz):
    freq_hz = scipy.fft.rfftfreq(
        waveform.values.size, d=1.0 / waveform.fs
    )
    amplitude = np.abs(scipy.fft.rfft(waveform.values, workers=-1))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        gridspec_kw={"width_ratios": (1.0, 3.2)},
        sharey=True,
        constrained_layout=True,
    )
    axes[0].plot(amplitude, freq_hz, color="#475569", lw=0.8)
    axes[0].set(
        xlabel="Amplitude",
        ylabel="Frequency (Hz)",
        title="Fourier spectrum",
    )
    image = axes[1].imshow(
        power_to_db(result["mean_tf_power"]),
        origin="lower",
        aspect="auto",
        extent=_frequency_extent(
            waveform, result["mean_tf_power"].shape[0]
        ),
        cmap="viridis",
        vmin=-80,
        vmax=0,
    )
    for mode in range(1, result["modes"].shape[0]):
        color = _mode_color(mode)
        low, high = result["frequency_bands_hz"][mode]
        axes[1].plot(
            waveform.time_s,
            result["center_freq_hz"][mode],
            color=color,
            lw=1.1,
            label=f"Mode {mode}: {low:.1f}-{high:.1f} Hz",
        )
        for boundary in (low, high):
            axes[0].axhline(boundary, color=color, ls="--", lw=0.8)
            axes[1].axhline(boundary, color=color, ls="--", lw=0.8)
    axes[1].axvline(0, color="white", ls=":", lw=0.8)
    axes[1].set(
        xlabel="Time (s)",
        title=f"{waveform.direction}: TF spectrum and IF tracks",
    )
    axes[1].legend(frameon=True, fontsize=8, loc="upper right")
    _limit_frequency_axis(axes[0], plot_max_hz, waveform.fs)
    _limit_frequency_axis(axes[1], plot_max_hz, waveform.fs)
    fig.colorbar(image, ax=axes[1], label="Relative power (dB)")
    return fig


def plot_single_waveform_results(waveform, result):
    return {
        "input_tf": plot_input_and_tf(waveform, result, PLOT_MAX_HZ),
        "modes": plot_modes(waveform, result),
        "if_reconstruction": plot_if_and_reconstruction(
            waveform, result, PLOT_MAX_HZ
        ),
        "spectrum_if_mapping": plot_spectrum_if_mapping(
            waveform, result, PLOT_MAX_HZ
        ),
    }
'''.strip()


ANALYSIS_AND_EXPORT = r'''
def analyze_single_waveform(waveform):
    x = waveform.values.reshape(1, -1)
    raw = run_dynamic_stvmd_batched(
        x,
        fs=waveform.fs,
        K=K,
        alpha=ALPHA,
        window_length=WINDOW_LENGTH,
        tau=TAU,
        tol=TOL,
        max_iters=MAX_ITERS,
        batch_windows=BATCH_WINDOWS,
    )
    return summarize_stvmd_result(x, waveform.fs, raw)


def save_single_waveform_results(output_dir, waveform, result, figures):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = waveform.direction.lower()
    for name, figure in figures.items():
        figure.savefig(
            output_dir / f"{prefix}_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
    np.savez_compressed(
        output_dir / "stvmd_single_waveform_results.npz",
        input_file=str(waveform.path),
        direction=waveform.direction,
        fs=waveform.fs,
        time_s=waveform.time_s,
        modes=result["modes"],
        center_freq_hz=result["center_freq_hz"],
        reconstruction=result["reconstruction"],
        nrmse=result["nrmse"],
        energy_fraction=result["energy_fraction"],
        frequency_bands_hz=result["frequency_bands_hz"],
    )
'''.strip()


LOAD_AND_RUN = r'''
waveform = load_single_waveform(INPUT_FILE, DIRECTION)
print("文件:", waveform.path.resolve())
print("方向:", waveform.direction)
print("采样率:", waveform.fs, "Hz")
print("样本数:", waveform.values.size)
print(
    "时间范围:",
    (float(waveform.time_s[0]), float(waveform.time_s[-1])),
    "s",
)
print("时窗时间:", WINDOW_LENGTH / waveform.fs, "s")
print("频率分辨率:", waveform.fs / WINDOW_LENGTH, "Hz")

result = analyze_single_waveform(waveform)
figures = plot_single_waveform_results(waveform, result)
for figure in figures.values():
    display(figure)
'''.strip()


SAVE = r'''
OUTPUT_DIR = Path("output/stvmd_single_waveform")
if SAVE_OUTPUTS:
    save_single_waveform_results(OUTPUT_DIR, waveform, result, figures)
    print(f"结果已保存到: {OUTPUT_DIR.resolve()}")
else:
    print("SAVE_OUTPUTS=False：未写出结果文件。")
'''.strip()


def build():
    cells = [
        markdown(
            "# 单波形分批动态 STVMD 分析\n\n"
            "读取一个 Instantel ASCII/TXT 文件，从 `Tran`、`Vert`、"
            "`Long` 中选择一个完整波形进行分批动态 STVMD。"
            "只需修改下一节的参数单元格。",
            "single-stvmd-00",
        ),
        markdown("## 1. 导入依赖", "single-stvmd-01"),
        code(IMPORTS, "single-stvmd-02", tags=("core",)),
        markdown(
            "## 2. 手动参数\n\n"
            "`WINDOW_LENGTH` 是短时时窗长度；`BATCH_WINDOWS` 只控制"
            "每批同时计算多少个窗口，不改变滑动步长。滑动步长固定为1。",
            "single-stvmd-03",
        ),
        code(CONFIG, "single-stvmd-04", tags=("parameters",)),
        markdown("## 3. 读取一个 TXT 波形", "single-stvmd-05"),
        code(LOADER, "single-stvmd-06", tags=("core",)),
        markdown("## 4. 分批动态 STVMD", "single-stvmd-07"),
        code(single_stvmd_source(), "single-stvmd-08", tags=("core",)),
        markdown("## 5. 结果诊断", "single-stvmd-09"),
        code(DIAGNOSTICS, "single-stvmd-10", tags=("core",)),
        markdown("## 6. 绘图与保存函数", "single-stvmd-11"),
        code(PLOTTING, "single-stvmd-12", tags=("core",)),
        code(ANALYSIS_AND_EXPORT, "single-stvmd-13", tags=("core",)),
        markdown("## 7. 运行完整波形分析", "single-stvmd-14"),
        code(LOAD_AND_RUN, "single-stvmd-15"),
        markdown("## 8. 保存结果", "single-stvmd-16"),
        code(SAVE, "single-stvmd-17"),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
