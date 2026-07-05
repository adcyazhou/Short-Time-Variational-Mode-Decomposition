from pathlib import Path

import nbformat
from nbformat import v4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"
OUTPUT_NOTEBOOK = ROOT / "single_waveform_vmd_stvmd_original.ipynb"


def find_code_cell(notebook, required_markers):
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and all(marker in cell.source for marker in required_markers)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source cell for {tuple(required_markers)}, "
            f"got {len(matches)}"
        )
    return matches[0]


def markdown(source, cell_id):
    return v4.new_markdown_cell(source=source, id=cell_id)


def code(source, cell_id, tags=()):
    cell = v4.new_code_cell(source=source, id=cell_id)
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


SOURCE = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
BUFFER_SOURCE = find_code_cell(
    SOURCE,
    ("def buffer(", "def unbuffer(", "def window_norm("),
)
VMD_SOURCE = find_code_cell(SOURCE, ("class VMD(object):",))
STVMD_SOURCE = find_code_cell(SOURCE, ("class STVMD(object):",))


IMPORTS = r'''
from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython.display import display
from numba import jit, prange
from scipy.fft import irfft, rfft
from tqdm import tqdm
'''.strip()


PARAMETERS = r'''
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 1000

VMD_N_FFT = 64
STVMD_WINDOW_LENGTH = 512

PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True
'''.strip()


LOADER = r'''
@dataclass(frozen=True)
class SingleWaveform:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    direction: str
    time_s: np.ndarray
    values: np.ndarray


def _header_metadata(lines):
    metadata = {}
    for line in lines:
        normalized = line.strip().strip("\"'")
        if ":" not in normalized:
            continue
        key, value = normalized.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _numeric_rows_after_header(lines):
    header_index = None
    required = {"Tran", "Vert", "Long"}
    for index, line in enumerate(lines):
        words = set(line.strip().strip("\"'").split())
        if required.issubset(words):
            header_index = index
            break
    if header_index is None:
        raise ValueError("Could not find Tran Vert Long data header")

    rows = []
    for line in lines[header_index + 1:]:
        fields = line.strip().strip("\"'").split()
        if len(fields) < 3:
            continue
        try:
            rows.append([float(value) for value in fields[:3]])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No numeric waveform rows found after data header")
    return np.asarray(rows, dtype=float)


def load_single_waveform(path, direction):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if direction not in {"Tran", "Vert", "Long"}:
        raise ValueError("direction must be one of Tran, Vert, or Long")

    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    metadata = _header_metadata(lines)
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    sample_rate_match = re.search(
        number_pattern, metadata.get("Sample Rate", "")
    )
    if sample_rate_match is None:
        raise ValueError("Sample Rate metadata is missing or invalid")
    fs = float(sample_rate_match.group())
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample Rate must be a finite positive number")

    pretrigger_match = re.search(
        number_pattern, metadata.get("Pre-trigger Length", "")
    )
    pretrigger_seconds = (
        abs(float(pretrigger_match.group()))
        if pretrigger_match is not None
        else 0.0
    )

    rows = _numeric_rows_after_header(lines)
    column = {"Tran": 0, "Vert": 1, "Long": 2}[direction]
    values = rows[:, column]
    if not np.isfinite(values).all():
        raise ValueError("Waveform values must all be finite")
    time_s = np.arange(values.size) / fs - pretrigger_seconds
    return SingleWaveform(
        path=path,
        metadata=metadata,
        fs=fs,
        pretrigger_seconds=pretrigger_seconds,
        direction=direction,
        time_s=time_s,
        values=values,
    )
'''.strip()


ANALYSIS_CORE = r'''
def validate_config(
    waveform,
    K,
    alpha,
    tau,
    tol,
    max_iters,
    vmd_n_fft,
    stvmd_window_length,
    plot_max_hz,
):
    if waveform.values.ndim != 1 or not np.isfinite(waveform.values).all():
        raise ValueError("Waveform must be a finite one-dimensional array")
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer not smaller than 2")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS must be an integer not smaller than 2")
    if not isinstance(vmd_n_fft, (int, np.integer)) or vmd_n_fft < 2:
        raise ValueError("VMD_N_FFT must be an integer not smaller than 2")
    if (
        not isinstance(stvmd_window_length, (int, np.integer))
        or stvmd_window_length < 2
        or stvmd_window_length > waveform.values.size
    ):
        raise ValueError(
            "STVMD_WINDOW_LENGTH must be between 2 and the sample count"
        )
    for name, value in (
        ("ALPHA", alpha),
        ("TAU", tau),
        ("TOL", tol),
        ("PLOT_MAX_HZ", plot_max_hz),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")


def estimate_vmd_memory_gb(channels, samples, K, n_fft, max_iters):
    padded = samples + n_fft - 1
    bins = padded // 2 + 1
    bytes_total = (
        max_iters * channels * bins * K * 16
        + max_iters * channels * bins * 16
        + max_iters * K * 8
    )
    return bytes_total / (1024 ** 3)


def estimate_stvmd_memory_gb(
    channels, samples, K, window_length, max_iters
):
    frames = samples
    bins = window_length // 2 + 1
    bytes_total = (
        2 * channels * bins * K * frames * 16
        + 2 * channels * bins * frames * 16
        + frames * channels * bins * 16
        + max_iters * K * frames * 8
    )
    return bytes_total / (1024 ** 3), frames


def run_original_vmd(
    x,
    fs,
    K=4,
    alpha=50.0,
    tau=1e-5,
    tol=1e-9,
    max_iters=1000,
    n_fft=64,
):
    x = np.asarray(x, dtype=float)
    model = VMD(
        num_channel=x.shape[0],
        n_fft=n_fft,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    f_hat = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(f_hat)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_frequency_hz": omega * (fs / 2.0),
        "dynamic": False,
    }


def run_original_stvmd(
    x,
    fs,
    K=4,
    alpha=50.0,
    tau=1e-5,
    tol=1e-9,
    max_iters=1000,
    window_length=512,
):
    x = np.asarray(x, dtype=float)
    window = scipy.signal.windows.hamming(
        window_length, sym=False
    )
    model = STVMD(
        num_channel=x.shape[0],
        n_fft=window_length,
        window_func=window,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    f_hat, windowed = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(f_hat, dynamic=True)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_frequency_hz": omega * (fs / 2.0),
        "windowed_signal": windowed,
        "dynamic": True,
        "hop_length": model.hop_len,
    }


def single_sided_amplitude(modes, fs):
    values = np.asarray(modes, dtype=float)[:, 0, :]
    sample_count = values.shape[-1]
    spectrum = np.fft.rfft(values, axis=-1)
    amplitude = np.abs(spectrum) / sample_count
    if sample_count % 2 == 0:
        amplitude[:, 1:-1] *= 2.0
    else:
        amplitude[:, 1:] *= 2.0
    frequency_hz = np.fft.rfftfreq(sample_count, d=1.0 / fs)
    return frequency_hz, amplitude


def modal_metrics(modes, fs):
    frequency_hz, amplitude = single_sided_amplitude(modes, fs)
    energy = np.sum(np.asarray(modes, dtype=float)[:, 0, :] ** 2, axis=1)
    total = float(np.sum(energy))
    energy_fraction = (
        energy / total
        if total > np.finfo(float).eps
        else np.zeros_like(energy)
    )
    return {
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "energy": energy,
        "energy_fraction": energy_fraction,
    }


def add_modal_metrics(raw_result, fs):
    result = dict(raw_result)
    result.update(modal_metrics(raw_result["modes"], fs))
    return result
'''.strip()


def build():
    notebook = v4.new_notebook(
        cells=[
            markdown(
                "# Single-waveform original VMD then dynamic STVMD",
                "title",
            ),
            code(IMPORTS, "imports", tags=("core",)),
            code(PARAMETERS, "parameters", tags=("parameters",)),
            code(LOADER, "loader", tags=("core",)),
            code(
                BUFFER_SOURCE,
                "original-buffer-source",
                tags=("core", "original-algorithm-source"),
            ),
            code(
                VMD_SOURCE,
                "original-vmd-source",
                tags=("core", "original-algorithm-source"),
            ),
            code(
                STVMD_SOURCE,
                "original-stvmd-source",
                tags=("core", "original-algorithm-source"),
            ),
            code(ANALYSIS_CORE, "analysis-core", tags=("core",)),
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
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    build()
