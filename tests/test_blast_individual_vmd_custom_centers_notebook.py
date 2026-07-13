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
