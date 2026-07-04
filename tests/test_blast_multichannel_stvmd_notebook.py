from pathlib import Path
import os
import subprocess
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import nbformat
import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def test_convert_instantel_ascii_to_csv_uses_pandas(tmp_path):
    ns = notebook_namespace()
    source = tmp_path / "sample.TXT"
    target = tmp_path / "data_csv" / "sample.csv"
    source.write_text(instantel_text(), encoding="utf-8")
    info = ns["convert_instantel_ascii_to_csv"](source, target)
    frame = pd.read_csv(target)
    assert info["fs"] == 4096
    assert info["pretrigger_seconds"] == 0.5
    assert frame.columns.tolist() == [
        "Sample", "Time_s", "Tran", "Vert", "Long"
    ]
    assert frame["Time_s"].iloc[0] == -0.5
    assert frame[["Tran", "Vert", "Long"]].to_numpy().tolist() == [
        [1, 2, 3], [4, 5, 6]
    ]


def test_generated_notebook_embeds_original_stvmd_source_verbatim():
    generated = nbformat.read(NOTEBOOK, as_version=4)
    source = nbformat.read(ROOT / "main_STVMD.ipynb", as_version=4)
    copied = [
        cell.source for cell in generated.cells
        if "original-stvmd" in cell.metadata.get("tags", [])
    ]
    assert copied == [source.cells[index].source for index in (0, 1, 3)]
    joined = "\n".join(copied)
    assert "@jit(nopython=True, cache=True)" in joined
    assert "self.hop_len = 1" in joined
    assert "tqdm" in joined
    assert "def apply_dynamic" in joined
    assert "run_dynamic_stvmd_batched" not in joined


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


def test_load_csv_direction_inputs_truncates_and_orders_channels(tmp_path):
    ns = notebook_namespace()
    paths = {}
    for distance, n, offset in (("5m", 7, 0), ("10m", 4, 100), ("15m", 6, 200)):
        frame = pd.DataFrame({
            "Sample": np.arange(n),
            "Time_s": np.arange(n) / 4096 - 0.5,
            "Tran": np.arange(n) + offset,
            "Vert": np.arange(n) + offset + 10,
            "Long": np.arange(n) + offset + 20,
        })
        paths[distance] = tmp_path / f"{distance}.csv"
        frame.to_csv(paths[distance], index=False)
    _, signals, time_s = ns["load_csv_direction_inputs"](paths)
    assert signals["Tran"].shape == (3, 4)
    np.testing.assert_array_equal(signals["Tran"][:, 0], [0, 100, 200])
    assert time_s[0] == -0.5


def test_original_stvmd_adapter_returns_expected_shapes():
    ns = notebook_namespace()
    x = synthetic_multichannel(n=64)
    result = ns["run_original_stvmd"](
        x, fs=128, K=3, alpha=50.0, window_length=16,
        tau=1e-5, tol=1e-5, max_iters=8,
    )
    assert result["modes"].shape == (3, 3, 64)
    assert result["center_freq_hz"].shape == (3, 64)
    assert result["mean_tf_power"].shape == (9, 64)
    assert np.isfinite(result["modes"]).all()


def test_summarize_result_reports_reconstruction_and_bands():
    ns = notebook_namespace()
    x = synthetic_multichannel(n=128)
    raw = ns["run_original_stvmd"](
        x, fs=128, K=3, alpha=50.0, window_length=16,
        tau=1e-5, tol=1e-5, max_iters=8,
    )
    summary = ns["summarize_stvmd_result"](x, 128, raw)
    assert summary["reconstruction"].shape == x.shape
    assert summary["nrmse"].shape == (3,)
    assert summary["energy_fraction"].shape == (3, 3)
    assert summary["frequency_bands_hz"].shape == (3, 2)
    assert np.all(
        summary["frequency_bands_hz"][:, 0]
        <= summary["frequency_bands_hz"][:, 1]
    )


def test_power_to_db_has_zero_db_maximum():
    ns = notebook_namespace()
    power = np.array([[1.0, 10.0], [100.0, 0.0]])
    db = ns["power_to_db"](power)
    assert np.max(db) == pytest.approx(0.0)
    assert np.isfinite(db).all()


def diagnostic_fixture(ns):
    x = synthetic_multichannel(n=128)
    raw = ns["run_original_stvmd"](
        x, fs=128, K=3, alpha=50.0, window_length=16,
        tau=1e-5, tol=1e-5, max_iters=8,
    )
    return x, ns["summarize_stvmd_result"](x, 128, raw)


def test_plot_functions_return_expected_axes():
    ns = notebook_namespace()
    x, result = diagnostic_fixture(ns)
    time_s = np.arange(x.shape[1]) / 128 - 0.5
    fig1 = ns["plot_input_and_tf"]("Tran", x, time_s, 128, result, 64)
    fig2 = ns["plot_modes"]("Tran", time_s, result)
    fig3 = ns["plot_if_and_reconstruction"](
        "Tran", x, time_s, result, 64
    )
    fig4 = ns["plot_spectrum_if_mapping"](
        "Tran", x, time_s, 128, result, 64
    )
    assert len(fig1.axes) >= 2
    assert len(fig2.axes) == 9
    assert len(fig3.axes) >= 3
    assert len(fig4.axes) >= 2
    for figure in (fig1, fig2, fig3, fig4):
        figure.canvas.draw()
        plt.close(figure)


def test_save_outputs_write_four_pngs_and_combined_npz(tmp_path):
    ns = notebook_namespace()
    x, result = diagnostic_fixture(ns)
    time_s = np.arange(x.shape[1]) / 128 - 0.5
    figures = {
        "input_tf": ns["plot_input_and_tf"](
            "Tran", x, time_s, 128, result, 64
        ),
        "modes": ns["plot_modes"]("Tran", time_s, result),
        "if_reconstruction": ns["plot_if_and_reconstruction"](
            "Tran", x, time_s, result, 64
        ),
        "spectrum_if_mapping": ns["plot_spectrum_if_mapping"](
            "Tran", x, time_s, 128, result, 64
        ),
    }
    ns["save_direction_figures"](tmp_path, "Tran", figures)
    ns["save_all_results"](
        tmp_path,
        {"Tran": result},
        {"K": 3, "ALPHA": 50.0, "WINDOW_LENGTH": 32},
    )
    assert (tmp_path / "stvmd_results.npz").is_file()
    assert len(list(tmp_path.glob("tran_*.png"))) == 4
    for figure in figures.values():
        plt.close(figure)


def test_notebook_executes_in_quick_test_mode():
    script = (
        "import os, nbformat;"
        "from nbclient import NotebookClient;"
        "os.environ['STVMD_QUICK_TEST']='1';"
        f"p=r'{NOTEBOOK}';"
        "nb=nbformat.read(p,as_version=4);"
        f"NotebookClient(nb,timeout=300,kernel_name='python3',"
        f"resources={{'metadata':{{'path':r'{ROOT}'}}}}).execute();"
        "print('NOTEBOOK_SMOKE_OK')"
    )
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=360,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NOTEBOOK_SMOKE_OK" in completed.stdout
