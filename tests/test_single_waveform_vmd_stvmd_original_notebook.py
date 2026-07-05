from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib
import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "single_waveform_vmd_stvmd_original.ipynb"
GENERATOR = (
    ROOT / "tools" / "build_single_waveform_vmd_stvmd_original_notebook.py"
)
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"


def joined_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_artifacts_exist():
    assert NOTEBOOK.is_file()
    assert GENERATOR.is_file()


def test_manual_parameters_have_one_source_of_truth():
    source = joined_source()
    for line in (
        'INPUT_FILE = Path("5m.TXT")',
        'DIRECTION = "Tran"',
        "K = 4",
        "ALPHA = 50.0",
        "TAU = 1e-5",
        "TOL = 1e-9",
        "MAX_ITERS = 1000",
        "VMD_N_FFT = 64",
        "STVMD_WINDOW_LENGTH = 512",
        "PLOT_MAX_HZ = 200.0",
        "SAVE_OUTPUTS = True",
    ):
        assert source.count(line) == 1


def test_notebook_is_independent_and_orders_vmd_before_stvmd():
    source = joined_source()
    for forbidden in (
        "single_waveform_vmd_original",
        "single_waveform_stvmd_batched",
        "blast_multichannel_stvmd",
    ):
        assert forbidden not in source
    assert source.index("vmd_result = run_original_vmd") < source.index(
        "stvmd_result = run_original_stvmd"
    )


def test_notebook_contains_required_pipeline_only():
    source = joined_source()
    for marker in (
        "def buffer(",
        "class VMD(object):",
        "class STVMD(object):",
        "def load_single_waveform",
        "def run_original_vmd",
        "def run_original_stvmd",
        "def single_sided_amplitude",
        "def modal_metrics",
        "def plot_modal_time",
        "def plot_modal_frequency",
        "def plot_energy_fraction",
        "def save_analysis",
        'OUTPUT_DIR = Path("output/vmd_stvmd_single_waveform")',
    ):
        assert marker in source
    for forbidden in (
        "instantaneous frequency",
        "time-frequency",
        "reconstruction error",
    ):
        assert forbidden not in source.lower()
