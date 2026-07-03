from pathlib import Path

import nbformat
import numpy as np
import pytest


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


def synthetic_multichannel(fs=128, n=256):
    t = np.arange(n) / fs
    base_1 = np.sin(2 * np.pi * 20 * t)
    base_2 = 0.6 * np.sin(2 * np.pi * 28 * t)
    return np.vstack(
        [
            base_1 + base_2,
            0.7 * base_1 + 1.1 * base_2,
            1.2 * base_1 + 0.5 * base_2,
        ]
    )


def test_validate_config_rejects_non_repository_window():
    ns = notebook_namespace()
    with pytest.raises(ValueError, match="WINDOW_LENGTH"):
        ns["validate_config"](3, 50.0, 512, 16, 50)


def test_batched_dynamic_stvmd_returns_finite_aligned_results():
    ns = notebook_namespace()
    x = synthetic_multichannel(n=128)
    result = ns["run_dynamic_stvmd_batched"](
        x,
        fs=128,
        K=3,
        alpha=50.0,
        window_length=32,
        tau=1e-5,
        tol=1e-6,
        max_iters=80,
        batch_windows=17,
    )
    assert result["modes"].shape == (3, 3, 128)
    assert result["center_freq_hz"].shape == (3, 128)
    assert result["mean_tf_power"].shape == (17, 128)
    assert np.isfinite(result["modes"]).all()
    assert np.isfinite(result["center_freq_hz"]).all()
    np.testing.assert_allclose(result["center_freq_hz"][0], 0.0)


def test_single_batch_and_split_batches_agree():
    ns = notebook_namespace()
    x = synthetic_multichannel(n=96)
    kwargs = dict(
        fs=128,
        K=3,
        alpha=50.0,
        window_length=32,
        tau=1e-5,
        tol=1e-7,
        max_iters=100,
    )
    whole = ns["run_dynamic_stvmd_batched"](x, batch_windows=96, **kwargs)
    split = ns["run_dynamic_stvmd_batched"](x, batch_windows=13, **kwargs)
    np.testing.assert_allclose(
        split["center_freq_hz"], whole["center_freq_hz"], atol=2e-3
    )
    np.testing.assert_allclose(split["modes"], whole["modes"], atol=2e-3)
