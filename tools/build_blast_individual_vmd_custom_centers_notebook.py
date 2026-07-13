"""Build the self-contained nine-signal custom-center VMD notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "blast_individual_vmd_custom_centers.ipynb"


def markdown(source, cell_id):
    return new_markdown_cell(source, id=cell_id)


def code(source, cell_id):
    return new_code_cell(source, id=cell_id)


TITLE = """# 5m/10m/15m 九通道自定义中心频率 VMD

算法更新方程来自 `figure_experiment_STVMD_ssvep_singlechannel.ipynb`。
用户中心仅作为初值，后续中心继续迭代；alpha 固定为 2000。
"""

IMPORTS = """from dataclasses import dataclass
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy.fft import irfft, rfft
"""

CONFIG = r'''
INPUT_FILES = {
    "5m": Path("5m.TXT"),
    "10m": Path("10m.TXT"),
    "15m": Path("15m.TXT"),
}

# 在这里分别修改九条信号的 K 和全部 K 个初始中心频率。
# 程序不会自动添加 0 Hz 模态；如需要，请在 centers_hz 中明确填写 0.0。
VMD_CONFIG = {
    distance: {
        direction: {"K": 3, "centers_hz": [10.0, 40.0, 100.0]}
        for direction in ("Tran", "Vert", "Long")
    }
    for distance in ("5m", "10m", "15m")
}

ALPHA = 2000.0
N_FFT = 64
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 10000
PLOT_DPI = 120

QUICK_TEST = os.environ.get("BLAST_VMD_QUICK_TEST") == "1"
if QUICK_TEST:
    MAX_ITERS = 20
'''.strip()

LOADER = r'''
@dataclass(frozen=True)
class BlastRecord:
    path: Path
    fs: float
    pretrigger_seconds: float
    unit: str
    channels: dict
    time_s: np.ndarray


def _metadata_number(metadata, key):
    if key not in metadata:
        raise ValueError(f"missing metadata: {key}")
    token = metadata[key].split()[0]
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"invalid {key}: {metadata[key]}") from exc


def load_instantel_record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    metadata, header_index = {}, None
    for index, raw in enumerate(lines):
        stripped = raw.strip().strip('"')
        if all(name in stripped for name in ("Tran", "Vert", "Long")):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()
    if header_index is None:
        raise ValueError(f"{path.name}: missing Tran/Vert/Long header")
    data = np.atleast_2d(
        np.loadtxt(lines[header_index + 1 :], dtype=float)
    )
    if (
        data.shape[1] != 3
        or data.shape[0] == 0
        or not np.isfinite(data).all()
    ):
        raise ValueError(f"{path.name}: expected finite three-column data")
    fs = _metadata_number(metadata, "Sample Rate")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{path.name}: sample rate must be positive")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    time_s = np.arange(data.shape[0], dtype=float) / fs - pretrigger
    channels = {
        name: data[:, index].copy()
        for index, name in enumerate(("Tran", "Vert", "Long"))
    }
    return BlastRecord(
        path=path,
        fs=fs,
        pretrigger_seconds=pretrigger,
        unit="mm/s",
        channels=channels,
        time_s=time_s,
    )
'''.strip()

VALIDATION = r'''
def validate_signal_config(distance, direction, config, fs):
    key = f"{distance}/{direction}"
    K = config.get("K")
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(f"{key}: K must be a positive integer")
    centers = np.asarray(config.get("centers_hz"), dtype=float)
    if centers.ndim != 1 or centers.size != K:
        raise ValueError(
            f"{key}: K={K} but centers_hz has {centers.size} values"
        )
    if not np.isfinite(centers).all():
        raise ValueError(f"{key}: centers_hz must be finite")
    if np.any(np.diff(centers) <= 0):
        raise ValueError(f"{key}: centers_hz must be strictly increasing")
    if np.any(centers < 0) or np.any(centers >= fs / 2.0):
        raise ValueError(
            f"{key}: centers_hz must lie below the Nyquist frequency"
        )
    return centers


def validate_global_config(alpha, n_fft, tau, tol, max_iters):
    if alpha != 2000.0:
        raise ValueError("ALPHA is fixed at 2000.0")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("N_FFT must be an integer >= 2")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("TAU must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("TOL must be finite and positive")
    if (
        not isinstance(max_iters, (int, np.integer))
        or max_iters < 2
    ):
        raise ValueError("MAX_ITERS must be an integer >= 2")
'''.strip()

WARM_START_VMD = r'''
def centers_hz_to_internal(centers_hz, fs):
    return np.asarray(centers_hz, dtype=float) / (fs / 2.0)


def centers_internal_to_hz(centers, fs):
    return np.asarray(centers, dtype=float) * (fs / 2.0)


class WarmStartVMD:
    """Source VMD equations with updateable user-supplied initial centers."""

    def __init__(
        self,
        num_channel,
        n_fft=64,
        alpha=2000.0,
        K=3,
        tol=1e-9,
        tau=1e-5,
        maxiters=10000,
    ):
        self.num_channel = num_channel
        self.n_fft = n_fft
        self.alpha = alpha * np.ones(K)
        self.K = K
        self.tol = tol
        self.tau = tau
        self.maxiters = maxiters
        self.padwidth = (
            ((n_fft - 1) // 2, (n_fft - 1) // 2)
            if (n_fft - 1) % 2 == 0
            else ((n_fft - 1) // 2 + 1, (n_fft - 1) // 2)
        )

    def prepare_offline(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[0] != self.num_channel:
            raise ValueError(
                "x must have shape (num_channel, samples)"
            )
        self.len_x = x.shape[1]
        padded = np.pad(x, ((0, 0), self.padwidth), mode="reflect")
        return rfft(padded, axis=1, workers=-1)

    def apply(self, f_hat_plus, omega_init):
        channels, frequency_bins = f_hat_plus.shape
        freqs = (
            np.arange(1, frequency_bins + 1, dtype=float)
            / frequency_bins
        )
        omega_init = np.asarray(omega_init, dtype=float)
        if omega_init.shape != (self.K,):
            raise ValueError(f"omega_init must contain {self.K} values")

        omega_state = np.zeros((2, self.K), dtype=float)
        omega_state[0] = omega_init
        u_state = np.zeros(
            (2, channels, frequency_bins, self.K), dtype=complex
        )
        lambda_state = np.zeros(
            (2, channels, frequency_bins), dtype=complex
        )
        sum_uk = np.zeros((channels, frequency_bins), dtype=complex)
        converged = False

        for iteration in range(self.maxiters - 1):
            current = iteration % 2
            next_index = (iteration + 1) % 2
            for k in range(self.K):
                previous_k = self.K - 1 if k == 0 else k - 1
                previous_state = current if k == 0 else next_index
                sum_uk = (
                    u_state[previous_state, :, :, previous_k]
                    + sum_uk
                    - u_state[current, :, :, k]
                )
                denominator = 1.0 + self.alpha[k] * (
                    freqs - omega_state[current, k]
                ) ** 2
                u_state[next_index, :, :, k] = (
                    f_hat_plus
                    - sum_uk
                    - lambda_state[current] / 2.0
                ) / denominator

                power = np.abs(u_state[next_index, :, :, k]) ** 2
                total_power = float(np.sum(power))
                if (
                    not np.isfinite(total_power)
                    or total_power <= np.finfo(float).eps
                ):
                    raise FloatingPointError(
                        f"mode {k + 1} has zero or invalid energy"
                    )
                omega_state[next_index, k] = np.sum(
                    freqs * np.sum(power, axis=0)
                ) / total_power

            lambda_state[next_index] = lambda_state[current] + self.tau * (
                np.sum(u_state[next_index], axis=2) - f_hat_plus
            )
            delta = u_state[next_index] - u_state[current]
            u_diff = float(np.mean(np.abs(delta) ** 2))
            if u_diff < self.tol and iteration > 2:
                converged = True
                break

        final_index = (iteration + 1) % 2
        order = np.argsort(omega_state[final_index])
        return (
            u_state[final_index][:, :, order],
            omega_state[final_index, order],
            iteration + 1,
            converged,
            order,
        )

    def postprocess(self, u_hat):
        padded_n = self.len_x + sum(self.padwidth)
        u = irfft(
            u_hat, n=padded_n, axis=1, workers=-1
        ).real
        u = np.transpose(u, (2, 0, 1))
        return u[
            :,
            :,
            self.padwidth[0] : padded_n - self.padwidth[1],
        ]
'''.strip()

ANALYSIS = r'''
def run_warm_start_vmd(
    signal,
    fs,
    K,
    centers_hz,
    alpha,
    n_fft,
    tau,
    tol,
    max_iters,
    data_key,
):
    distance, direction = data_key
    signal = np.asarray(signal, dtype=float)
    if (
        signal.ndim != 1
        or signal.size < 2
        or not np.isfinite(signal).all()
    ):
        raise ValueError(
            f"{distance}/{direction}: signal must be finite and one-dimensional"
        )
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{distance}/{direction}: fs must be positive")
    validate_global_config(alpha, n_fft, tau, tol, max_iters)
    centers = validate_signal_config(
        distance,
        direction,
        {"K": K, "centers_hz": centers_hz},
        fs,
    )
    model = WarmStartVMD(
        1,
        n_fft=n_fft,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    spectrum = model.prepare_offline(signal.reshape(1, -1))
    mode_spectrum, omega, iterations, converged, order = model.apply(
        spectrum, centers_hz_to_internal(centers, fs)
    )
    modes = model.postprocess(mode_spectrum)[:, 0, :]
    reconstruction = np.sum(modes, axis=0)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "initial_centers_hz": centers[order],
        "final_centers_hz": centers_internal_to_hz(omega, fs),
        "reconstruction": reconstruction,
        "reconstruction_rmse": float(
            np.sqrt(np.mean((signal - reconstruction) ** 2))
        ),
        "iterations": iterations,
        "converged": converged,
    }


def analyze_all_records(
    records, config, alpha, n_fft, tau, tol, max_iters
):
    results = {}
    for distance in ("5m", "10m", "15m"):
        record = records[distance]
        for direction in ("Tran", "Vert", "Long"):
            item = config[distance][direction]
            result = run_warm_start_vmd(
                record.channels[direction],
                fs=record.fs,
                K=item["K"],
                centers_hz=item["centers_hz"],
                alpha=alpha,
                n_fft=n_fft,
                tau=tau,
                tol=tol,
                max_iters=max_iters,
                data_key=(distance, direction),
            )
            results[(distance, direction)] = result
    return results


def print_vmd_summary(distance, direction, record, result):
    print(
        f"{distance}/{direction}: samples={record.time_s.size}, "
        f"fs={record.fs:g} Hz"
    )
    print(
        "  initial centers (Hz):",
        np.array2string(result["initial_centers_hz"], precision=3),
    )
    print(
        "  final centers (Hz):  ",
        np.array2string(result["final_centers_hz"], precision=3),
    )
    print(
        f"  iterations={result['iterations']}, "
        f"converged={result['converged']}"
    )
    print(
        f"  reconstruction RMSE="
        f"{result['reconstruction_rmse']:.6g}"
    )
'''.strip()

PLOTTING = r'''
def plot_vmd_modes(
    distance,
    direction,
    time_s,
    signal,
    result,
    alpha=2000.0,
):
    modes = result["modes"]
    figure, axes = plt.subplots(
        modes.shape[0] + 1,
        1,
        figsize=(12, 2.0 * (modes.shape[0] + 1)),
        sharex=True,
        constrained_layout=True,
        dpi=120,
    )
    axes[0].plot(time_s, signal, color="#202020", linewidth=0.8)
    axes[0].set_ylabel("Original\n(mm/s)")
    axes[0].grid(alpha=0.2)
    for index, mode in enumerate(modes):
        axis = axes[index + 1]
        axis.plot(time_s, mode, color="#0072B2", linewidth=0.75)
        axis.set_ylabel(f"Mode {index + 1}\n(mm/s)")
        axis.set_title(
            f"init={result['initial_centers_hz'][index]:.2f} Hz, "
            f"final={result['final_centers_hz'][index]:.2f} Hz",
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Relative time (s)")
    figure.suptitle(
        f"{distance} {direction}: K={modes.shape[0]}, "
        f"alpha={alpha:g}"
    )
    return figure
'''.strip()

LOAD_RECORDS = r'''
records = {
    distance: load_instantel_record(path)
    for distance, path in INPUT_FILES.items()
}
'''.strip()

RUN_ALL = r'''
results = analyze_all_records(
    records, VMD_CONFIG, ALPHA, N_FFT, TAU, TOL, MAX_ITERS
)
figures = {}
for distance in ("5m", "10m", "15m"):
    for direction in ("Tran", "Vert", "Long"):
        key = (distance, direction)
        print_vmd_summary(
            distance, direction, records[distance], results[key]
        )
        figures[key] = plot_vmd_modes(
            distance,
            direction,
            records[distance].time_s,
            records[distance].channels[direction],
            results[key],
            alpha=ALPHA,
        )
        display(figures[key])
'''.strip()

PLACEHOLDER = "pass"


def build():
    notebook = new_notebook(
        cells=[
            markdown(TITLE, "title"),
            code(IMPORTS, "imports"),
            code(CONFIG, "config"),
            code(LOADER, "loader"),
            code(VALIDATION, "validation"),
            code(WARM_START_VMD, "warm-start-vmd"),
            code(ANALYSIS, "analysis"),
            code(PLOTTING, "plotting"),
            code(LOAD_RECORDS, "load-records"),
            code(RUN_ALL, "run-all"),
        ],
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


if __name__ == "__main__":
    build()
