from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from build_single_waveform_stvmd_batched_notebook import LOADER


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"
TARGET = ROOT / "single_waveform_vmd_original.ipynb"


def find_code_cell(notebook, required_markers):
    matches = [
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
        and all(marker in "".join(cell.source) for marker in required_markers)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source cell for {required_markers}, "
            f"got {len(matches)}"
        )
    return matches[0]


def markdown(source, cell_id):
    return new_markdown_cell(source, id=cell_id)


def code(source, cell_id, tags=()):
    return new_code_cell(
        source,
        id=cell_id,
        metadata={"tags": list(tags)} if tags else {},
    )


IMPORTS = r'''
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython.display import display
from numba import jit, prange
from scipy.fft import irfft, rfft
from tqdm import tqdm
'''.strip()


CONFIG = r'''
# 只在这里修改输入文件、方向和VMD参数。
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
N_FFT = 64
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 10000

PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True

print(
    f"file={INPUT_FILE}, direction={DIRECTION}, K={K}, "
    f"alpha={ALPHA}, n_fft={N_FFT}, max_iters={MAX_ITERS}"
)
'''.strip()


VMD_ADAPTER = r'''
def validate_vmd_config(K, alpha, n_fft, tau, tol, max_iters):
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K 必须为不小于2的整数")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("N_FFT 必须为不小于2的整数")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS 必须为不小于2的整数")
    for name, value in (
        ("ALPHA", alpha),
        ("TAU", tau),
        ("TOL", tol),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")


def estimate_original_vmd_memory_gb(
    channels, samples, K, n_fft, max_iters
):
    left = n_fft // 2
    right = n_fft - 1 - left
    padded_samples = samples + left + right
    frequency_bins = padded_samples // 2 + 1
    complex_bytes = 16
    float_bytes = 8
    u_bytes = (
        max_iters * channels * frequency_bins * K * complex_bytes
    )
    lambda_bytes = (
        max_iters * channels * frequency_bins * complex_bytes
    )
    omega_bytes = max_iters * K * float_bytes
    return (u_bytes + lambda_bytes + omega_bytes) / (1024 ** 3)


def run_original_vmd(
    x,
    fs,
    K=4,
    alpha=50.0,
    n_fft=64,
    tau=1e-5,
    tol=1e-9,
    max_iters=10000,
):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("输入必须为有限的 (通道, 时间) 二维数组")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数")
    validate_vmd_config(K, alpha, n_fft, tau, tol, max_iters)
    model = VMD(
        num_channel=x.shape[0],
        n_fft=n_fft,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    spectrum = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(spectrum)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_freq_hz": omega * (fs / 2.0),
    }
'''.strip()


DIAGNOSTICS = r'''
def _energy_band(freq_hz, power, low=0.05, high=0.95):
    power = np.maximum(np.asarray(power, dtype=float), 0.0)
    total = float(np.sum(power))
    if total <= np.finfo(float).eps:
        return np.array([0.0, 0.0])
    cumulative = np.cumsum(power) / total
    return np.array(
        [
            np.interp(low, cumulative, freq_hz),
            np.interp(high, cumulative, freq_hz),
        ]
    )


def summarize_vmd_result(x, fs, raw):
    modes = raw["modes"]
    reconstruction = np.sum(modes, axis=0)
    denominator = np.linalg.norm(x, axis=1)
    nrmse = np.divide(
        np.linalg.norm(x - reconstruction, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    mode_energy = np.sum(modes ** 2, axis=2)
    energy_total = np.sum(mode_energy, axis=0, keepdims=True)
    energy_fraction = np.divide(
        mode_energy,
        energy_total,
        out=np.zeros_like(mode_energy),
        where=energy_total > np.finfo(float).eps,
    )
    frequency_hz = scipy.fft.rfftfreq(
        modes.shape[-1], d=1.0 / fs
    )
    mode_power = np.zeros((modes.shape[0], frequency_hz.size))
    frequency_bands_hz = np.zeros((modes.shape[0], 2))
    for mode in range(modes.shape[0]):
        spectra = scipy.fft.rfft(modes[mode], axis=1, workers=-1)
        mode_power[mode] = np.sum(np.abs(spectra) ** 2, axis=0)
        frequency_bands_hz[mode] = _energy_band(
            frequency_hz, mode_power[mode]
        )
    result = dict(raw)
    result.update(
        {
            "reconstruction": reconstruction,
            "nrmse": nrmse,
            "energy_fraction": energy_fraction,
            "frequency_hz": frequency_hz,
            "mode_power": mode_power,
            "frequency_bands_hz": frequency_bands_hz,
        }
    )
    return result
'''.strip()


PLOTTING = r'''
def _mode_color(mode):
    return plt.get_cmap("tab10")(mode % 10)


def _component_label(mode):
    return "Residual" if mode == 0 else f"Mode {mode}"


def plot_input_and_modes(waveform, result):
    modes = result["modes"][:, 0, :]
    fig, axes = plt.subplots(
        modes.shape[0] + 1,
        1,
        figsize=(11, 2.0 * (modes.shape[0] + 1)),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        waveform.time_s, waveform.values, color="#475569", lw=0.8
    )
    axes[0].set_ylabel("Input\nVelocity (mm/s)")
    axes[0].set_title(
        f"{waveform.path.name} {waveform.direction}: original VMD"
    )
    for mode in range(modes.shape[0]):
        axes[mode + 1].plot(
            waveform.time_s,
            modes[mode],
            color=_mode_color(mode),
            lw=0.75,
        )
        axes[mode + 1].set_ylabel(
            f"{_component_label(mode)}\nVelocity (mm/s)"
        )
    for axis in axes:
        axis.axvline(0, color="black", ls="--", lw=0.6)
    axes[-1].set_xlabel("Time (s)")
    return fig


def plot_center_frequencies(waveform, result, plot_max_hz):
    center = result["center_freq_hz"]
    labels = [_component_label(mode) for mode in range(center.size)]
    fig, axis = plt.subplots(
        1, 1, figsize=(10, 4.2), constrained_layout=True
    )
    bars = axis.bar(
        np.arange(center.size),
        center,
        color=[_mode_color(mode) for mode in range(center.size)],
    )
    axis.set(
        xlabel="Component",
        ylabel="Center frequency (Hz)",
        title="Constant VMD center frequencies",
        ylim=(0.0, min(float(plot_max_hz), waveform.fs / 2.0)),
    )
    axis.set_xticks(np.arange(center.size), labels)
    axis.bar_label(bars, fmt="%.2f Hz", fontsize=9)
    return fig


def plot_reconstruction_and_energy(waveform, result):
    mode_n = result["modes"].shape[0]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": (1.4, 1.0)},
        constrained_layout=True,
    )
    axes[0].plot(
        waveform.time_s,
        waveform.values,
        color="#64748b",
        lw=0.8,
        label="Input",
    )
    axes[0].plot(
        waveform.time_s,
        result["reconstruction"][0],
        color="#D55E00",
        lw=0.7,
        label="Reconstruction",
    )
    axes[0].axvline(0, color="black", ls="--", lw=0.6)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Velocity (mm/s)",
        title=f"Reconstruction  NRMSE={result['nrmse'][0]:.3g}",
    )
    axes[0].legend(frameon=False)
    fractions = result["energy_fraction"][:, 0]
    bars = axes[1].bar(
        np.arange(mode_n),
        fractions,
        color=[_mode_color(mode) for mode in range(mode_n)],
    )
    axes[1].set(
        xlabel="Component",
        ylabel="Energy fraction",
        title="Component energy fraction",
    )
    axes[1].set_xticks(
        np.arange(mode_n),
        [_component_label(mode) for mode in range(mode_n)],
    )
    axes[1].bar_label(bars, fmt="%.3f", fontsize=9)
    return fig


def plot_mode_spectra(waveform, result, plot_max_hz):
    mode_n = result["modes"].shape[0]
    fig, axis = plt.subplots(
        1, 1, figsize=(11, 5.2), constrained_layout=True
    )
    for mode in range(mode_n):
        low, high = result["frequency_bands_hz"][mode]
        axis.plot(
            result["frequency_hz"],
            result["mode_power"][mode],
            color=_mode_color(mode),
            lw=0.9,
            label=(
                f"{_component_label(mode)}: "
                f"{low:.1f}-{high:.1f} Hz"
            ),
        )
        axis.axvline(low, color=_mode_color(mode), ls=":", lw=0.6)
        axis.axvline(high, color=_mode_color(mode), ls=":", lw=0.6)
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="Power",
        title="VMD component spectra",
        xlim=(0.0, min(float(plot_max_hz), waveform.fs / 2.0)),
    )
    axis.legend(frameon=False, fontsize=8)
    return fig


def plot_vmd_results(waveform, result):
    return {
        "input_modes": plot_input_and_modes(waveform, result),
        "center_frequencies": plot_center_frequencies(
            waveform, result, PLOT_MAX_HZ
        ),
        "reconstruction_energy": plot_reconstruction_and_energy(
            waveform, result
        ),
        "mode_spectra": plot_mode_spectra(
            waveform, result, PLOT_MAX_HZ
        ),
    }
'''.strip()


ANALYSIS_AND_EXPORT = r'''
def analyze_single_waveform_vmd(waveform):
    raw = run_original_vmd(
        waveform.values.reshape(1, -1),
        fs=waveform.fs,
        K=K,
        alpha=ALPHA,
        n_fft=N_FFT,
        tau=TAU,
        tol=TOL,
        max_iters=MAX_ITERS,
    )
    return summarize_vmd_result(
        waveform.values.reshape(1, -1), waveform.fs, raw
    )


def save_vmd_results(output_dir, waveform, result, figures):
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
        output_dir / "vmd_single_waveform_results.npz",
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
estimated_memory_gb = estimate_original_vmd_memory_gb(
    1, waveform.values.size, K, N_FFT, MAX_ITERS
)
print("文件:", waveform.path.resolve())
print("方向:", waveform.direction)
print("采样率:", waveform.fs, "Hz")
print("样本数:", waveform.values.size)
print(
    "时间范围:",
    (float(waveform.time_s[0]), float(waveform.time_s[-1])),
    "s",
)
print(f"原始VMD粗略内存估算: {estimated_memory_gb:.2f} GB")
if estimated_memory_gb > 2.0:
    warnings.warn(
        "当前参数的原始VMD内存估算超过2 GB；"
        "请考虑降低MAX_ITERS或K。",
        RuntimeWarning,
    )

result = analyze_single_waveform_vmd(waveform)
figures = plot_vmd_results(waveform, result)
for figure in figures.values():
    display(figure)
'''.strip()


SAVE = r'''
OUTPUT_DIR = Path("output/vmd_single_waveform")
if SAVE_OUTPUTS:
    save_vmd_results(OUTPUT_DIR, waveform, result, figures)
    print(f"结果已保存到: {OUTPUT_DIR.resolve()}")
else:
    print("SAVE_OUTPUTS=False：未写出结果文件。")
'''.strip()


def build():
    source_notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    buffer_source = find_code_cell(
        source_notebook,
        ("def buffer(", "def unbuffer(", "def window_norm("),
    )
    vmd_source = find_code_cell(
        source_notebook,
        ("class VMD(object):",),
    )
    cells = [
        markdown(
            "# 单波形原始 VMD 分析\n\n"
            "读取一个 Instantel ASCII/TXT 文件，从 `Tran`、`Vert`、"
            "`Long` 中选择一个完整波形，并使用原仓库 VMD 源码进行"
            "整段稳态分解。",
            "single-vmd-00",
        ),
        markdown("## 1. 导入依赖", "single-vmd-01"),
        code(IMPORTS, "single-vmd-02", tags=("core",)),
        markdown(
            "## 2. 手动参数\n\n"
            "`N_FFT` 沿用原仓库参数名，用于反射填充宽度，"
            "不是STVMD滑动时窗。",
            "single-vmd-03",
        ),
        code(CONFIG, "single-vmd-04", tags=("parameters",)),
        markdown("## 3. 读取一个 TXT 波形", "single-vmd-05"),
        code(LOADER, "single-vmd-06", tags=("core",)),
        markdown(
            "## 4. 原仓库缓冲函数与 VMD 类\n\n"
            "下面两个单元格逐字来自 `main_STVMD.ipynb`。",
            "single-vmd-07",
        ),
        code(
            buffer_source,
            "single-vmd-08",
            tags=("core", "original-vmd-source"),
        ),
        code(
            vmd_source,
            "single-vmd-09",
            tags=("core", "original-vmd-source"),
        ),
        markdown("## 5. VMD适配、诊断和绘图", "single-vmd-10"),
        code(VMD_ADAPTER, "single-vmd-11", tags=("core",)),
        code(DIAGNOSTICS, "single-vmd-12", tags=("core",)),
        code(PLOTTING, "single-vmd-13", tags=("core",)),
        code(ANALYSIS_AND_EXPORT, "single-vmd-14", tags=("core",)),
        markdown("## 6. 运行完整波形 VMD", "single-vmd-15"),
        code(LOAD_AND_RUN, "single-vmd-16"),
        markdown("## 7. 保存结果", "single-vmd-17"),
        code(SAVE, "single-vmd-18"),
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
