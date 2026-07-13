from pathlib import Path
import os
import subprocess
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT / "tools" / "build_blast_individual_vmd_custom_centers_notebook.py"
)
NOTEBOOK = ROOT / "blast_individual_vmd_custom_centers.ipynb"


def notebook_namespace(*cell_ids):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    for cell_id in cell_ids:
        cell = next(cell for cell in notebook.cells if cell.id == cell_id)
        exec(compile(cell.source, f"<{cell_id}>", "exec"), namespace)
    return namespace


def write_instantel(path, fs=128, rows=((1, 2, 3), (4, 5, 6))):
    body = "\n".join(" ".join(map(str, row)) for row in rows)
    path.write_text(
        f'"Sample Rate : {fs} sps"\n'
        '"Pre-trigger Length : -0.500 sec"\n'
        '"Units : mm/s and "\n'
        "Tran Vert Long\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_builder_generates_ordered_self_contained_notebook():
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert [cell.id for cell in notebook.cells] == [
        "title",
        "imports",
        "config",
        "loader",
        "validation",
        "warm-start-vmd",
        "analysis",
        "plotting",
        "load-records",
        "run-all",
    ]
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "figure_experiment_STVMD_ssvep_singlechannel.ipynb" in source
    assert "ALPHA = 2000.0" in source
    assert "VMD_CONFIG" in source
    assert "from tools." not in source


def test_config_contains_independent_entries_for_all_nine_signals():
    ns = notebook_namespace("imports", "config")
    assert ns["ALPHA"] == 2000.0
    assert set(ns["VMD_CONFIG"]) == {"5m", "10m", "15m"}
    for distance in ("5m", "10m", "15m"):
        assert set(ns["VMD_CONFIG"][distance]) == {
            "Tran",
            "Vert",
            "Long",
        }
        for config in ns["VMD_CONFIG"][distance].values():
            assert config["K"] == len(config["centers_hz"])


def test_load_instantel_record_preserves_all_three_channels(tmp_path):
    path = tmp_path / "record.TXT"
    write_instantel(path, fs=256)
    ns = notebook_namespace("imports", "loader")
    record = ns["load_instantel_record"](path)
    assert record.fs == 256
    assert record.pretrigger_seconds == 0.5
    np = ns["np"]
    np.testing.assert_allclose(record.channels["Tran"], [1, 4])
    np.testing.assert_allclose(record.channels["Vert"], [2, 5])
    np.testing.assert_allclose(record.channels["Long"], [3, 6])
    np.testing.assert_allclose(
        record.time_s, [-0.5, -0.5 + 1 / 256]
    )
