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

VMD_BUGGY_DELTA = "delta = u_hat_plus[0,:,i]-u_hat_plus[1,:,i]"
VMD_CORRECTED_DELTA = "delta = u_hat_plus[0,:,:,i]-u_hat_plus[1,:,:,i]"


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
        "PLOT_MAX_HZ = None",
        "SAVE_OUTPUTS = True",
    ):
        assert source.count(line) == 1


def test_algorithm_sources_preserve_only_the_audited_vmd_correction():
    generated = nbformat.read(NOTEBOOK, as_version=4)
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    generated_buffer = find_source_cell(
        generated, ("def buffer(", "def unbuffer(", "def window_norm(")
    )
    generated_vmd = find_source_cell(generated, ("class VMD(object):",))
    generated_stvmd = find_source_cell(generated, ("class STVMD(object):",))
    source_buffer = find_source_cell(
        source, ("def buffer(", "def unbuffer(", "def window_norm(")
    )
    source_vmd = find_source_cell(source, ("class VMD(object):",))
    source_stvmd = find_source_cell(source, ("class STVMD(object):",))
    assert source_vmd.count(VMD_BUGGY_DELTA) == 1
    expected_vmd = source_vmd.replace(
        VMD_BUGGY_DELTA, VMD_CORRECTED_DELTA
    )
    assert generated_buffer == source_buffer
    assert generated_vmd == expected_vmd
    assert generated_stvmd == source_stvmd


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


def test_original_adapters_return_full_length_modes():
    namespace = notebook_namespace()
    fs = 128.0
    time_s = np.arange(64) / fs
    values = np.sin(2 * np.pi * 20 * time_s).reshape(1, -1)
    common = dict(
        fs=fs,
        K=3,
        alpha=50.0,
        tau=1e-5,
        tol=1e-4,
        max_iters=4,
    )
    vmd = namespace["run_original_vmd"](
        values, n_fft=16, **common
    )
    stvmd = namespace["run_original_stvmd"](
        values, window_length=16, **common
    )
    assert vmd["modes"].shape == (3, 1, 64)
    assert vmd["center_frequency_hz"].shape == (3,)
    assert stvmd["modes"].shape == (3, 1, 64)
    assert stvmd["center_frequency_hz"].shape == (3, 64)
    assert stvmd["dynamic"] is True
    assert stvmd["hop_length"] == 1


def test_corrected_vmd_recovers_known_20_and_28_hz_components():
    namespace = notebook_namespace()
    fs = 128.0
    time_s = np.arange(256) / fs
    values = (
        np.sin(2 * np.pi * 20 * time_s)
        + 0.5 * np.sin(2 * np.pi * 28 * time_s)
    ).reshape(1, -1)

    result = namespace["run_original_vmd"](
        values,
        fs=fs,
        K=3,
        alpha=50.0,
        tau=1e-5,
        tol=1e-9,
        max_iters=1000,
        n_fft=64,
    )
    result = namespace["add_modal_metrics"](result, fs)

    np.testing.assert_allclose(
        result["center_frequency_hz"][1:],
        [20.0, 28.0],
        atol=1.0,
    )
    peaks = result["frequency_hz"][
        np.argmax(result["amplitude"][1:], axis=1)
    ]
    np.testing.assert_allclose(peaks, [20.0, 28.0], atol=1.0)


def test_single_sided_amplitude_has_physical_units():
    namespace = notebook_namespace()
    fs = 128.0
    time_s = np.arange(128) / fs
    modes = np.zeros((2, 1, 128))
    modes[0, 0] = 2.0 * np.sin(2 * np.pi * 10 * time_s)
    frequency_hz, amplitude = namespace["single_sided_amplitude"](
        modes, fs
    )
    peak = np.argmin(np.abs(frequency_hz - 10.0))
    assert amplitude[0, peak] == pytest.approx(2.0)


def test_modal_metrics_include_residual_and_sum_to_one():
    namespace = notebook_namespace()
    modes = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0, 2.0]],
        ]
    )
    metrics = namespace["modal_metrics"](modes, fs=8.0)
    np.testing.assert_allclose(metrics["energy"], [4.0, 16.0])
    np.testing.assert_allclose(metrics["energy_fraction"], [0.2, 0.8])
    assert metrics["energy_fraction"].sum() == pytest.approx(1.0)


def synthetic_result(namespace):
    fs = 128.0
    time_s = np.arange(64) / fs - 0.25
    modes = np.stack(
        [
            0.2 * np.ones(64),
            np.sin(2 * np.pi * 10 * (time_s + 0.25)),
            0.5 * np.sin(2 * np.pi * 20 * (time_s + 0.25)),
        ]
    )[:, None, :]
    result = namespace["modal_metrics"](modes, fs)
    result.update(
        {
            "modes": modes,
            "center_frequency_hz": np.array([0.0, 10.0, 20.0]),
        }
    )
    return time_s, fs, result


def test_validate_config_allows_automatic_plot_limit_and_rejects_bad_limits(
    tmp_path,
):
    namespace = notebook_namespace()
    waveform = namespace["SingleWaveform"](
        path=tmp_path / "single.TXT",
        metadata={},
        fs=128.0,
        pretrigger_seconds=0.0,
        direction="Tran",
        time_s=np.arange(64) / 128.0,
        values=np.ones(64),
    )
    namespace["validate_config"](
        waveform,
        K=3,
        alpha=50.0,
        tau=1e-5,
        tol=1e-9,
        max_iters=1000,
        vmd_n_fft=64,
        stvmd_window_length=16,
        plot_max_hz=None,
    )
    with pytest.raises(
        ValueError,
        match="PLOT_MAX_HZ must be None or a finite positive number",
    ):
        namespace["validate_config"](
            waveform,
            K=3,
            alpha=50.0,
            tau=1e-5,
            tol=1e-9,
            max_iters=1000,
            vmd_n_fft=64,
            stvmd_window_length=16,
            plot_max_hz=0.0,
        )


def test_vmd_center_frequency_plot_uses_horizontal_lines():
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    figure = namespace["plot_center_frequencies"](
        "VMD", time_s, result, plot_max_hz=None
    )
    expected = result["center_frequency_hz"]
    for axis, frequency_hz in zip(figure.axes, expected):
        line = axis.lines[0]
        np.testing.assert_allclose(line.get_xdata(), time_s)
        np.testing.assert_allclose(
            line.get_ydata(),
            np.full(time_s.shape, frequency_hz),
        )
    matplotlib.pyplot.close(figure)


def test_stvmd_center_frequency_plot_preserves_dynamic_tracks():
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    result["center_frequency_hz"] = np.vstack(
        [
            np.zeros(time_s.size),
            np.linspace(9.0, 11.0, time_s.size),
            np.linspace(18.0, 22.0, time_s.size),
        ]
    )
    figure = namespace["plot_center_frequencies"](
        "STVMD", time_s, result, plot_max_hz=None
    )
    for axis, expected in zip(figure.axes, result["center_frequency_hz"]):
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
    matplotlib.pyplot.close(figure)


def test_automatic_center_frequency_limits_do_not_hide_high_modes():
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    result["center_frequency_hz"] = np.array([0.0, 50.0, 215.0, 1378.0])
    result["modes"] = np.zeros((4, 1, time_s.size))
    result["energy_fraction"] = np.full(4, 0.25)
    figure = namespace["plot_center_frequencies"](
        "VMD", time_s, result, plot_max_hz=None
    )
    for axis, frequency_hz in zip(
        figure.axes, result["center_frequency_hz"]
    ):
        lower, upper = axis.get_ylim()
        assert lower <= frequency_hz <= upper
    matplotlib.pyplot.close(figure)


def test_each_method_has_exactly_three_requested_figures():
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    figures = namespace["plot_method_results"](
        "VMD", time_s, fs, result, plot_max_hz=50.0
    )
    assert set(figures) == {
        "time_modes",
        "center_frequencies",
        "energy_fraction",
    }
    assert len(figures["time_modes"].axes) == 3
    assert len(figures["center_frequencies"].axes) == 3
    assert len(figures["energy_fraction"].axes) == 1
    for figure in figures.values():
        matplotlib.pyplot.close(figure)


def test_save_analysis_writes_six_pngs_and_npz(tmp_path):
    namespace = notebook_namespace()
    time_s, fs, vmd_result = synthetic_result(namespace)
    _, _, stvmd_result = synthetic_result(namespace)
    waveform = namespace["SingleWaveform"](
        path=tmp_path / "single.TXT",
        metadata={},
        fs=fs,
        pretrigger_seconds=0.25,
        direction="Tran",
        time_s=time_s,
        values=np.sum(vmd_result["modes"][:, 0], axis=0),
    )
    vmd_figures = namespace["plot_method_results"](
        "VMD", time_s, fs, vmd_result, 50.0
    )
    stvmd_figures = namespace["plot_method_results"](
        "STVMD", time_s, fs, stvmd_result, 50.0
    )
    namespace["save_analysis"](
        tmp_path,
        waveform,
        vmd_result,
        stvmd_result,
        vmd_figures,
        stvmd_figures,
        parameters={"K": 3},
    )
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "vmd_time_modes.png",
        "vmd_center_frequencies.png",
        "vmd_energy_fraction.png",
        "stvmd_time_modes.png",
        "stvmd_center_frequencies.png",
        "stvmd_energy_fraction.png",
    }
    saved = np.load(
        tmp_path / "vmd_stvmd_single_waveform_results.npz"
    )
    assert saved["vmd_modes"].shape == (3, 1, 64)
    assert saved["stvmd_modes"].shape == (3, 1, 64)
    assert saved["vmd_energy_fraction"].sum() == pytest.approx(1.0)
    assert saved["stvmd_energy_fraction"].sum() == pytest.approx(1.0)
    for figure in (*vmd_figures.values(), *stvmd_figures.values()):
        matplotlib.pyplot.close(figure)


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
            "corrected-vmd-source",
            ["core", "corrected-algorithm-source"],
        ),
        (
            "original-stvmd-source",
            ["core", "original-algorithm-source"],
        ),
        ("analysis-core", ["core"]),
        ("plotting", ["core"]),
        ("saving", ["core"]),
        ("load-heading", []),
        ("load-and-validate", []),
        ("vmd-heading", []),
        ("run-vmd", []),
        ("stvmd-heading", []),
        ("run-stvmd", []),
        ("export-heading", []),
        ("export", []),
        ("synthetic-stvmd-heading", []),
        ("synthetic-stvmd-two-sines", []),
    ]


def test_synthetic_two_sine_stvmd_example_is_appended():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert notebook.cells[-2].id == "synthetic-stvmd-heading"
    assert notebook.cells[-1].id == "synthetic-stvmd-two-sines"
    source = notebook.cells[-1].source
    for marker in (
        "t = np.arange(SYNTHETIC_SAMPLE_COUNT) / SYNTHETIC_FS",
        "x1 = np.sin(2 * np.pi * 20 * t)",
        "x2 = 0.5 * np.sin(2 * np.pi * 28 * t)",
        "synthetic_x = np.vstack([x1, x2])",
        "synthetic_stvmd_result = run_original_stvmd(",
        "plot_center_frequencies(",
    ):
        assert marker in source


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
        "def plot_center_frequencies",
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


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_notebook_executes_vmd_then_dynamic_stvmd(tmp_path):
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
    original_bytes = NOTEBOOK.read_bytes()
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
TAU = 1e-5
TOL = 1e-4
MAX_ITERS = 4
VMD_N_FFT = 16
STVMD_WINDOW_LENGTH = 16
PLOT_MAX_HZ = None
SAVE_OUTPUTS = False
'''.strip()
    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    executed_source = "\n".join(
        "".join(cell.source) for cell in executed.cells
    )
    assert executed_source.index("vmd_result = run_original_vmd") < (
        executed_source.index("stvmd_result = run_original_stvmd")
    )
    assert NOTEBOOK.read_bytes() == original_bytes
