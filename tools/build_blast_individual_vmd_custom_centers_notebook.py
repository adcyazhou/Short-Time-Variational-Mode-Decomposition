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

PLACEHOLDER = "pass"


def build():
    notebook = new_notebook(
        cells=[
            markdown(TITLE, "title"),
            code(IMPORTS, "imports"),
            code(CONFIG, "config"),
            code(LOADER, "loader"),
            code(PLACEHOLDER, "validation"),
            code(PLACEHOLDER, "warm-start-vmd"),
            code(PLACEHOLDER, "analysis"),
            code(PLACEHOLDER, "plotting"),
            code(PLACEHOLDER, "load-records"),
            code(PLACEHOLDER, "run-all"),
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
