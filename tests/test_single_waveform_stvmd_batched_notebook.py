from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient


matplotlib.use("Agg")


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


def instantel_text(rows, fs=128, pretrigger=0.5):
    body = "\n".join(f"{a} {b} {c}" for a, b, c in rows)
    return (
        f"Sample Rate: {fs} Hz\n"
        f"Pre-trigger Length: {pretrigger} sec\n"
        "Tran Vert Long\n"
        f"{body}\n"
    )


def test_load_single_waveform_selects_requested_direction(tmp_path):
    path = tmp_path / "single.TXT"
    path.write_text(
        instantel_text([(1, 10, 100), (2, 20, 200)]),
        encoding="utf-8",
    )
    namespace = notebook_namespace()
    waveform = namespace["load_single_waveform"](path, "Vert")
    np.testing.assert_array_equal(waveform.values, [10.0, 20.0])
    assert waveform.fs == 128
    assert waveform.time_s[0] == -0.5


def test_load_single_waveform_rejects_unknown_direction(tmp_path):
    path = tmp_path / "single.TXT"
    path.write_text(instantel_text([(1, 2, 3)]), encoding="utf-8")
    namespace = notebook_namespace()
    with pytest.raises(ValueError, match="DIRECTION"):
        namespace["load_single_waveform"](path, "Radial")


def test_window_validation_accepts_any_integer_length():
    namespace = notebook_namespace()
    namespace["validate_config"](2, 50.0, 7, 4, 3, 1e-5, 1e-4)
    with pytest.raises(ValueError, match="WINDOW_LENGTH"):
        namespace["validate_config"](2, 50.0, 1, 4, 3, 1e-5, 1e-4)


def test_batched_analysis_returns_single_channel_shapes():
    namespace = notebook_namespace()
    t = np.arange(64) / 128
    x = np.sin(2 * np.pi * 15 * t).reshape(1, -1)
    result = namespace["run_dynamic_stvmd_batched"](
        x,
        fs=128,
        K=3,
        alpha=50.0,
        window_length=16,
        tau=1e-5,
        tol=1e-4,
        max_iters=4,
        batch_windows=8,
    )
    assert result["modes"].shape == (3, 1, 64)
    assert result["center_freq_hz"].shape == (3, 64)


def test_plotting_and_saving_single_waveform_results(tmp_path):
    namespace = notebook_namespace()
    time_s = np.arange(64) / 128 - 0.25
    values = np.sin(2 * np.pi * 15 * (time_s + 0.25))
    waveform = namespace["SingleWaveform"](
        path=tmp_path / "single.TXT",
        metadata={},
        fs=128.0,
        pretrigger_seconds=0.25,
        direction="Tran",
        time_s=time_s,
        values=values,
    )
    raw = namespace["run_dynamic_stvmd_batched"](
        values.reshape(1, -1),
        fs=128,
        K=3,
        alpha=50.0,
        window_length=16,
        tau=1e-5,
        tol=1e-4,
        max_iters=4,
        batch_windows=8,
    )
    summary = namespace["summarize_stvmd_result"](
        values.reshape(1, -1), 128, raw
    )
    namespace["PLOT_MAX_HZ"] = 64.0
    figures = namespace["plot_single_waveform_results"](waveform, summary)
    assert set(figures) == {
        "input_tf",
        "modes",
        "if_reconstruction",
        "spectrum_if_mapping",
    }
    namespace["save_single_waveform_results"](
        tmp_path, waveform, summary, figures
    )
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert (tmp_path / "stvmd_single_waveform_results.npz").is_file()
    for figure in figures.values():
        matplotlib.pyplot.close(figure)


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_notebook_executes_with_patched_manual_parameters(tmp_path):
    rows = [
        (
            np.sin(2 * np.pi * 8 * index / 128),
            np.sin(2 * np.pi * 12 * index / 128),
            np.sin(2 * np.pi * 20 * index / 128),
        )
        for index in range(64)
    ]
    input_path = tmp_path / "single.TXT"
    input_path.write_text(instantel_text(rows), encoding="utf-8")
    notebook_hash = NOTEBOOK.read_bytes()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameter_cells = [
        cell
        for cell in notebook.cells
        if "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1
    parameter_cells[0].source = f'''
INPUT_FILE = Path({str(input_path)!r})
DIRECTION = "Tran"
K = 3
ALPHA = 50.0
WINDOW_LENGTH = 8
TAU = 1e-5
TOL = 1e-4
MAX_ITERS = 3
BATCH_WINDOWS = 4
PLOT_MAX_HZ = 64.0
SAVE_OUTPUTS = False
'''.strip()
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    assert NOTEBOOK.read_bytes() == notebook_hash
