from pathlib import Path
import os
import subprocess
import sys

import nbformat
import pytest


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


@pytest.mark.parametrize(
    "config, message",
    [
        ({"K": 3, "centers_hz": [10, 20]}, "K=3"),
        (
            {"K": 3, "centers_hz": [10, 10, 20]},
            "strictly increasing",
        ),
        (
            {"K": 3, "centers_hz": [20, 10, 30]},
            "strictly increasing",
        ),
        ({"K": 2, "centers_hz": [10, 64]}, "Nyquist"),
    ],
)
def test_validate_signal_config_rejects_invalid_centers(config, message):
    ns = notebook_namespace("imports", "validation")
    with pytest.raises(ValueError, match=message):
        ns["validate_signal_config"]("5m", "Tran", config, fs=128)


def test_validate_signal_config_does_not_insert_zero_center():
    ns = notebook_namespace("imports", "validation")
    centers = ns["validate_signal_config"](
        "5m",
        "Tran",
        {"K": 2, "centers_hz": [10, 20]},
        fs=128,
    )
    ns["np"].testing.assert_allclose(centers, [10, 20])


def test_hz_normalization_maps_nyquist_to_one():
    ns = notebook_namespace("imports", "validation", "warm-start-vmd")
    ns["np"].testing.assert_allclose(
        ns["centers_hz_to_internal"]([0, 32], 128), [0, 0.5]
    )
    ns["np"].testing.assert_allclose(
        ns["centers_internal_to_hz"]([0, 0.5], 128), [0, 32]
    )


def test_warm_start_vmd_updates_centers_and_reconstructs_two_sines():
    ns = notebook_namespace(
        "imports", "validation", "warm-start-vmd", "analysis"
    )
    np = ns["np"]
    fs = 128.0
    time_s = np.arange(512) / fs
    signal = np.sin(2 * np.pi * 20 * time_s) + 0.6 * np.sin(
        2 * np.pi * 28 * time_s
    )
    result = ns["run_warm_start_vmd"](
        signal,
        fs=fs,
        K=2,
        centers_hz=[17.0, 32.0],
        alpha=2000.0,
        n_fft=64,
        tau=1e-5,
        tol=1e-7,
        max_iters=100,
        data_key=("synthetic", "x"),
    )
    assert result["modes"].shape == (2, signal.size)
    assert np.isfinite(result["modes"]).all()
    assert not np.allclose(result["final_centers_hz"], [17.0, 32.0])
    input_rms = np.sqrt(np.mean(signal**2))
    assert result["reconstruction_rmse"] < 0.2 * input_rms
    assert result["iterations"] <= 100


def test_plot_vmd_modes_creates_original_plus_one_axis_per_mode():
    ns = notebook_namespace("imports", "plotting")
    np = ns["np"]
    time_s = np.arange(10) / 10
    signal = np.arange(10, dtype=float)
    result = {
        "modes": np.vstack([signal, -signal]),
        "initial_centers_hz": np.array([10.0, 20.0]),
        "final_centers_hz": np.array([11.0, 19.0]),
    }
    figure = ns["plot_vmd_modes"](
        "5m", "Tran", time_s, signal, result, alpha=2000.0
    )
    assert len(figure.axes) == 3
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert labels == [
        "Original\n(mm/s)",
        "Mode 1\n(mm/s)",
        "Mode 2\n(mm/s)",
    ]
    assert (
        "init=10.00 Hz, final=11.00 Hz"
        in figure.axes[1].get_title()
    )
    assert "alpha=2000" in figure._suptitle.get_text()
    ns["plt"].close(figure)


def test_analyze_all_records_runs_every_distance_direction_pair():
    ns = notebook_namespace(
        "imports", "loader", "validation", "analysis"
    )
    np = ns["np"]
    calls = []

    def fake_run(signal, **kwargs):
        calls.append(kwargs["data_key"])
        return {"modes": np.zeros((kwargs["K"], len(signal)))}

    ns["run_warm_start_vmd"] = fake_run
    record = ns["BlastRecord"](
        path=Path("x"),
        fs=128.0,
        pretrigger_seconds=0.5,
        unit="mm/s",
        channels={
            name: np.ones(16) for name in ("Tran", "Vert", "Long")
        },
        time_s=np.arange(16) / 128 - 0.5,
    )
    records = {
        distance: record for distance in ("5m", "10m", "15m")
    }
    config = {
        distance: {
            direction: {"K": 1, "centers_hz": [10.0]}
            for direction in ("Tran", "Vert", "Long")
        }
        for distance in records
    }
    results = ns["analyze_all_records"](
        records, config, 2000.0, 64, 1e-5, 1e-9, 20
    )
    assert len(results) == 9
    assert calls == [
        (distance, direction)
        for distance in ("5m", "10m", "15m")
        for direction in ("Tran", "Vert", "Long")
    ]
