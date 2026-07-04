from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "single_waveform_stvmd_batched.ipynb"
GENERATOR = ROOT / "tools" / "build_single_waveform_stvmd_batched_notebook.py"


def joined_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def notebook_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "core" in cell.metadata.get("tags", []):
            exec("".join(cell.source), namespace)
    return namespace


def test_single_waveform_notebook_and_generator_exist():
    assert NOTEBOOK.is_file()
    assert GENERATOR.is_file()


def test_manual_parameter_cell_has_one_source_of_truth():
    source = joined_source()
    for line in (
        'INPUT_FILE = Path("5m.TXT")',
        'DIRECTION = "Tran"',
        "K = 4",
        "ALPHA = 50.0",
        "WINDOW_LENGTH = 64",
        "TAU = 1e-5",
        "TOL = 1e-5",
        "MAX_ITERS = 300",
        "BATCH_WINDOWS = 128",
        "PLOT_MAX_HZ = 200.0",
        "SAVE_OUTPUTS = True",
    ):
        assert line in source
    assert "STVMD_QUICK_TEST" not in source
    assert "REPOSITORY_WINDOWS" not in source


def test_notebook_contains_single_waveform_pipeline():
    source = joined_source()
    for marker in (
        "def load_single_waveform",
        "def run_dynamic_stvmd_batched",
        "def analyze_single_waveform",
        "def plot_single_waveform_results",
        'OUTPUT_DIR = Path("output/stvmd_single_waveform")',
    ):
        assert marker in source
