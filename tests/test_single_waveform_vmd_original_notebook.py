from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "single_waveform_vmd_original.ipynb"
GENERATOR = ROOT / "tools" / "build_single_waveform_vmd_original_notebook.py"
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"


def joined_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_vmd_notebook_and_generator_exist():
    assert NOTEBOOK.is_file()
    assert GENERATOR.is_file()


def test_manual_parameters_have_one_source_of_truth():
    source = joined_source()
    for line in (
        'INPUT_FILE = Path("5m.TXT")',
        'DIRECTION = "Tran"',
        "K = 4",
        "ALPHA = 50.0",
        "N_FFT = 64",
        "TAU = 1e-5",
        "TOL = 1e-9",
        "MAX_ITERS = 10000",
        "PLOT_MAX_HZ = 200.0",
        "SAVE_OUTPUTS = True",
    ):
        assert source.count(line) == 1
    assert "STVMD_QUICK_TEST" not in source


def test_notebook_contains_original_vmd_pipeline():
    source = joined_source()
    for marker in (
        "class VMD(object):",
        "def load_single_waveform",
        "def run_original_vmd",
        "def summarize_vmd_result",
        "def plot_vmd_results",
        'OUTPUT_DIR = Path("output/vmd_single_waveform")',
    ):
        assert marker in source
