from pathlib import Path
import os
import subprocess
import sys

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


def find_source_cell(notebook, markers):
    matches = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and all(marker in cell.source for marker in markers)
    ]
    assert len(matches) == 1
    return matches[0]


def notebook_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if "core" not in cell.metadata.get("tags", []):
            continue
        exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    return namespace


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


def test_original_algorithm_sources_are_verbatim():
    generated = nbformat.read(NOTEBOOK, as_version=4)
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    generated_sources = [
        cell.source
        for cell in generated.cells
        if cell.cell_type == "code"
        and "original-algorithm-source"
        in cell.metadata.get("tags", [])
    ]
    expected = [
        find_source_cell(
            source,
            ("def buffer(", "def unbuffer(", "def window_norm("),
        ),
        find_source_cell(source, ("class VMD(object):",)),
        find_source_cell(source, ("class STVMD(object):",)),
    ]
    assert generated_sources == expected


def instantel_text(rows, fs=128, pretrigger=0.5):
    lines = [
        f'"Sample Rate: {fs} Hz"',
        f'"Pre-trigger Length: {-pretrigger} seconds"',
        '"Tran Vert Long"',
    ]
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


def test_fresh_loader_selects_direction_and_aligns_trigger(tmp_path):
    path = tmp_path / "waveform.TXT"
    path.write_text(
        instantel_text([(1, 10, 100), (2, 20, 200)]),
        encoding="utf-8",
    )

    waveform = notebook_namespace()["load_single_waveform"](path, "Vert")

    np.testing.assert_array_equal(waveform.values, [10.0, 20.0])
    assert waveform.fs == 128.0
    assert waveform.time_s[0] == pytest.approx(-0.5)
    assert waveform.direction == "Vert"


@pytest.mark.parametrize("fs", [0, -128, "9" * 400])
def test_fresh_loader_rejects_invalid_sample_rate(tmp_path, fs):
    path = tmp_path / "invalid-rate.TXT"
    path.write_text(
        instantel_text([(1, 10, 100)], fs=fs),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite positive"):
        notebook_namespace()["load_single_waveform"](path, "Vert")


def test_generator_regeneration_preserves_notebook_contract():
    before = NOTEBOOK.read_bytes()

    subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert NOTEBOOK.read_bytes() == before
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert [
        (cell.id, cell.metadata.get("tags", []))
        for cell in notebook.cells
    ] == [
        ("title", []),
        ("imports", ["core"]),
        ("parameters", ["parameters"]),
        ("loader", ["core"]),
        (
            "original-buffer-source",
            ["core", "original-algorithm-source"],
        ),
        (
            "original-vmd-source",
            ["core", "original-algorithm-source"],
        ),
        (
            "original-stvmd-source",
            ["core", "original-algorithm-source"],
        ),
    ]


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
