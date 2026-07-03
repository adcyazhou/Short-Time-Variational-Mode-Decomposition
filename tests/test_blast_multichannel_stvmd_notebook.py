from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "blast_multichannel_stvmd.ipynb"


def notebook_namespace():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns = {"__name__": "notebook_test"}
    for cell in nb.cells:
        if cell.cell_type == "code" and "core" in cell.metadata.get("tags", []):
            exec(compile(cell.source, str(NOTEBOOK), "exec"), ns)
    return ns


def instantel_text(fs=4096, rows=((1, 2, 3), (4, 5, 6))):
    body = "\n".join("\t".join(map(str, row)) for row in rows)
    return (
        '"Event Type : Full Waveform"\n'
        '"Event Time : 15:41:05"\n'
        '"Event Date : June 26, 2026"\n'
        '"Pre-trigger Length : -0.500 sec"\n'
        f'"Sample Rate : {fs} sps"\n'
        "\n   Tran   \t   Vert   \t   Long   \n"
        f"{body}\n"
    )


def test_notebook_contains_required_sections():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for heading in (
        "参数配置",
        "读取与校验",
        "动态多通道 STVMD",
        "Tran 方向",
        "Vert 方向",
        "Long 方向",
        "结果保存",
    ):
        assert heading in markdown


def test_load_instantel_txt_parses_metadata_and_columns(tmp_path):
    ns = notebook_namespace()
    path = tmp_path / "sample.TXT"
    path.write_text(instantel_text(), encoding="utf-8")
    record = ns["load_instantel_txt"](path)
    assert record.fs == 4096
    assert record.pretrigger_seconds == 0.5
    assert record.columns == ("Tran", "Vert", "Long")
    np.testing.assert_array_equal(
        record.data, np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    )


def test_prepare_direction_inputs_truncates_to_common_length(tmp_path):
    ns = notebook_namespace()
    records = {}
    for distance, n in (("5m", 7), ("10m", 4), ("15m", 6)):
        path = tmp_path / f"{distance}.TXT"
        path.write_text(
            instantel_text(rows=[(i, i + 10, i + 20) for i in range(n)]),
            encoding="utf-8",
        )
        records[distance] = ns["load_instantel_txt"](path)
    signals, time_s = ns["prepare_direction_inputs"](records)
    assert signals["Tran"].shape == (3, 4)
    np.testing.assert_array_equal(signals["Tran"][:, 0], [0, 0, 0])
    assert time_s[0] == -0.5
