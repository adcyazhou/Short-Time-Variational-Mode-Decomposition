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

CONFIG = "ALPHA = 2000.0\nVMD_CONFIG = {}"
PLACEHOLDER = "pass"


def build():
    notebook = new_notebook(
        cells=[
            markdown(TITLE, "title"),
            code(IMPORTS, "imports"),
            code(CONFIG, "config"),
            code(PLACEHOLDER, "loader"),
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
