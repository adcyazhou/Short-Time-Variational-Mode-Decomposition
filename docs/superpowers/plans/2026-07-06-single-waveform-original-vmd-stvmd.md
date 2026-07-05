# Single-waveform Original VMD and STVMD Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an independent notebook that analyzes one complete Instantel waveform with the repository's verbatim VMD followed by verbatim dynamic STVMD and produces only modal time histories, modal amplitude spectra, and modal energy fractions.

**Architecture:** A deterministic generator extracts the buffer helpers, `VMD` class, and `STVMD` class directly from `main_STVMD.ipynb` and embeds them unchanged. Newly written notebook-owned code handles TXT loading, validation, memory estimates, adapters, shared modal metrics, six figures, and optional NPZ/PNG saving.

**Tech Stack:** Python, Jupyter, nbformat, nbclient, NumPy, SciPy, Numba, tqdm, Matplotlib, pytest

---

### Task 1: Define the independent notebook contract

**Files:**
- Create: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Create later: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Create later: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Write the missing-artifact and parameter contract tests**

Create:

```python
from pathlib import Path
import os

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
```

- [ ] **Step 2: Write the source and output marker tests**

Append:

```python
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
    for forbidden_plot in (
        "instantaneous frequency",
        "time-frequency",
        "reconstruction error",
    ):
        assert forbidden_plot not in source.lower()
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py -q
```

Expected: failures because the notebook and generator do not exist.

- [ ] **Step 4: Commit the failing contract**

```powershell
git add -- tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "test: define original VMD STVMD notebook contract"
```

### Task 2: Generate a standalone notebook with verbatim algorithms and a fresh loader

**Files:**
- Create: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`

- [ ] **Step 1: Add exact-source test helpers**

Append to the test file:

```python
def find_source_cell(notebook, markers):
    matches = [
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
        and all(marker in "".join(cell.source) for marker in markers)
    ]
    assert len(matches) == 1
    return matches[0]


def notebook_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "core" in cell.metadata.get("tags", []):
            exec(
                compile("".join(cell.source), str(NOTEBOOK), "exec"),
                namespace,
            )
    return namespace


def test_original_algorithm_sources_are_verbatim():
    generated = nbformat.read(NOTEBOOK, as_version=4)
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    generated_sources = [
        "".join(cell.source)
        for cell in generated.cells
        if "original-algorithm-source" in cell.metadata.get("tags", [])
    ]
    expected = [
        find_source_cell(
            source, ("def buffer(", "def unbuffer(", "def window_norm(")
        ),
        find_source_cell(source, ("class VMD(object):",)),
        find_source_cell(source, ("class STVMD(object):",)),
    ]
    assert generated_sources == expected
```

- [ ] **Step 2: Add a synthetic Instantel loader test**

Append:

```python
def instantel_text(rows, fs=128, pretrigger=0.5):
    body = "\n".join(f"{tran} {vert} {long}" for tran, vert, long in rows)
    return (
        f'"Sample Rate : {fs} sps"\n'
        f'"Pre-trigger Length : -{pretrigger} sec"\n'
        '"Tran Vert Long"\n'
        f"{body}\n"
    )


def test_fresh_loader_selects_direction_and_aligns_trigger(tmp_path):
    path = tmp_path / "single.TXT"
    path.write_text(
        instantel_text([(1, 10, 100), (2, 20, 200)]),
        encoding="utf-8",
    )
    namespace = notebook_namespace()
    waveform = namespace["load_single_waveform"](path, "Vert")
    np.testing.assert_array_equal(waveform.values, [10.0, 20.0])
    assert waveform.fs == 128.0
    assert waveform.time_s[0] == pytest.approx(-0.5)
    assert waveform.direction == "Vert"
```

- [ ] **Step 3: Implement deterministic source extraction and cell helpers**

Create the generator with:

```python
from pathlib import Path

import nbformat
from nbformat import v4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"
OUTPUT_NOTEBOOK = ROOT / "single_waveform_vmd_stvmd_original.ipynb"


def find_code_cell(notebook, required_markers):
    matches = [
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
        and all(marker in "".join(cell.source) for marker in required_markers)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source cell for {required_markers}, got {len(matches)}"
        )
    return matches[0]


def markdown(source, cell_id):
    return v4.new_markdown_cell(source=source, id=cell_id)


def code(source, cell_id, tags=()):
    cell = v4.new_code_cell(source=source, id=cell_id)
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


source_notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
BUFFER_SOURCE = find_code_cell(
    source_notebook, ("def buffer(", "def unbuffer(", "def window_norm(")
)
VMD_SOURCE = find_code_cell(source_notebook, ("class VMD(object):",))
STVMD_SOURCE = find_code_cell(source_notebook, ("class STVMD(object):",))
```

Do not call `.strip()`, perform replacements, or format the three extracted
strings.

- [ ] **Step 4: Add imports and the sole parameter cell**

Use these generated code strings:

```python
IMPORTS = r'''
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython.display import display
from numba import jit, prange
from scipy.fft import irfft, rfft
from tqdm import tqdm
'''.strip()


PARAMETERS = r'''
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 1000

VMD_N_FFT = 64
STVMD_WINDOW_LENGTH = 512

PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True
'''.strip()
```

- [ ] **Step 5: Add a newly written TXT loader**

Add this generated core cell; do not import loader code from another project
file or notebook:

```python
LOADER = r'''
@dataclass(frozen=True)
class SingleWaveform:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    direction: str
    time_s: np.ndarray
    values: np.ndarray


def _header_metadata(lines):
    metadata = {}
    for line in lines:
        clean = line.strip().strip('"')
        if ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _numeric_rows_after_header(lines):
    header_index = None
    for index, line in enumerate(lines):
        words = line.replace('"', " ").split()
        if {"Tran", "Vert", "Long"}.issubset(words):
            header_index = index
            break
    if header_index is None:
        raise ValueError("Cannot find the Tran/Vert/Long data header")
    rows = []
    for line in lines[header_index + 1 :]:
        fields = line.replace('"', " ").split()
        if len(fields) < 3:
            continue
        try:
            rows.append(tuple(float(value) for value in fields[:3]))
        except ValueError:
            continue
    if not rows:
        raise ValueError("No numeric waveform samples were found")
    return np.asarray(rows, dtype=float)


def load_single_waveform(path, direction):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if direction not in {"Tran", "Vert", "Long"}:
        raise ValueError("DIRECTION must be Tran, Vert, or Long")
    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    metadata = _header_metadata(lines)
    sample_rate_text = metadata.get("Sample Rate", "")
    sample_rate_match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", sample_rate_text
    )
    if sample_rate_match is None:
        raise ValueError("Cannot parse Sample Rate")
    fs = float(sample_rate_match.group())
    pretrigger_text = metadata.get("Pre-trigger Length", "0")
    pretrigger_match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", pretrigger_text
    )
    pretrigger_seconds = (
        abs(float(pretrigger_match.group())) if pretrigger_match else 0.0
    )
    samples = _numeric_rows_after_header(lines)
    column = {"Tran": 0, "Vert": 1, "Long": 2}[direction]
    values = samples[:, column]
    if not np.isfinite(values).all():
        raise ValueError("Waveform contains non-finite values")
    time_s = np.arange(values.size) / fs - pretrigger_seconds
    return SingleWaveform(
        path=path,
        metadata=metadata,
        fs=fs,
        pretrigger_seconds=pretrigger_seconds,
        direction=direction,
        time_s=time_s,
        values=values,
    )
'''.strip()
```

- [ ] **Step 6: Build the initial notebook in the required order**

Implement `build()` with fixed cell IDs:

```python
def build():
    notebook = v4.new_notebook(
        cells=[
            markdown(
                "# Single-waveform original VMD then dynamic STVMD",
                "title",
            ),
            code(IMPORTS, "imports", tags=("core",)),
            code(PARAMETERS, "parameters", tags=("parameters",)),
            code(LOADER, "loader", tags=("core",)),
            code(
                BUFFER_SOURCE,
                "original-buffer-source",
                tags=("core", "original-algorithm-source"),
            ),
            code(
                VMD_SOURCE,
                "original-vmd-source",
                tags=("core", "original-algorithm-source"),
            ),
            code(
                STVMD_SOURCE,
                "original-stvmd-source",
                tags=("core", "original-algorithm-source"),
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    build()
```

- [ ] **Step 7: Generate and run the focused tests**

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "artifacts or manual_parameters or original_algorithm_sources or fresh_loader" `
  -q
```

Expected: artifact, parameter, source-identity, and loader tests pass. Pipeline
tests still fail because adapters and plots are not implemented.

- [ ] **Step 8: Commit the generator foundation**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "feat: scaffold original VMD STVMD notebook"
```

### Task 3: Add solver adapters and shared modal metrics

**Files:**
- Modify: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Write failing adapter shape tests**

Append:

```python
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
```

- [ ] **Step 2: Write failing spectrum and energy tests**

Append:

```python
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
```

- [ ] **Step 3: Run the tests and verify RED**

```powershell
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "adapters or single_sided or modal_metrics" -q
```

Expected: failures because the adapter and metric functions do not exist.

- [ ] **Step 4: Add validation and memory estimates**

Add a generated core string:

```python
ANALYSIS_CORE = r'''
def validate_config(
    waveform,
    K,
    alpha,
    tau,
    tol,
    max_iters,
    vmd_n_fft,
    stvmd_window_length,
    plot_max_hz,
):
    if waveform.values.ndim != 1 or not np.isfinite(waveform.values).all():
        raise ValueError("Waveform must be a finite one-dimensional array")
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer not smaller than 2")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS must be an integer not smaller than 2")
    if not isinstance(vmd_n_fft, (int, np.integer)) or vmd_n_fft < 2:
        raise ValueError("VMD_N_FFT must be an integer not smaller than 2")
    if (
        not isinstance(stvmd_window_length, (int, np.integer))
        or stvmd_window_length < 2
        or stvmd_window_length > waveform.values.size
    ):
        raise ValueError(
            "STVMD_WINDOW_LENGTH must be between 2 and the sample count"
        )
    for name, value in (
        ("ALPHA", alpha),
        ("TAU", tau),
        ("TOL", tol),
        ("PLOT_MAX_HZ", plot_max_hz),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")


def estimate_vmd_memory_gb(channels, samples, K, n_fft, max_iters):
    padded = samples + n_fft - 1
    bins = padded // 2 + 1
    bytes_total = (
        max_iters * channels * bins * K * 16
        + max_iters * channels * bins * 16
        + max_iters * K * 8
    )
    return bytes_total / (1024 ** 3)


def estimate_stvmd_memory_gb(
    channels, samples, K, window_length, max_iters
):
    frames = samples
    bins = window_length // 2 + 1
    bytes_total = (
        2 * channels * bins * K * frames * 16
        + 2 * channels * bins * frames * 16
        + frames * channels * bins * 16
        + max_iters * K * frames * 8
    )
    return bytes_total / (1024 ** 3), frames
'''.strip()
```

- [ ] **Step 5: Add the verbatim-algorithm adapters**

Continue `ANALYSIS_CORE` with:

```python
def run_original_vmd(
    x,
    fs,
    K=4,
    alpha=50.0,
    tau=1e-5,
    tol=1e-9,
    max_iters=1000,
    n_fft=64,
):
    x = np.asarray(x, dtype=float)
    model = VMD(
        num_channel=x.shape[0],
        n_fft=n_fft,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    f_hat = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(f_hat)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_frequency_hz": omega * (fs / 2.0),
        "dynamic": False,
    }


def run_original_stvmd(
    x,
    fs,
    K=4,
    alpha=50.0,
    tau=1e-5,
    tol=1e-9,
    max_iters=1000,
    window_length=512,
):
    x = np.asarray(x, dtype=float)
    window = scipy.signal.windows.hamming(
        window_length, sym=False
    )
    model = STVMD(
        num_channel=x.shape[0],
        n_fft=window_length,
        window_func=window,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    f_hat, windowed = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(f_hat, dynamic=True)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_frequency_hz": omega * (fs / 2.0),
        "windowed_signal": windowed,
        "dynamic": True,
        "hop_length": model.hop_len,
    }
```

- [ ] **Step 6: Add correct modal spectrum and energy calculations**

Continue `ANALYSIS_CORE` with:

```python
def single_sided_amplitude(modes, fs):
    values = np.asarray(modes, dtype=float)[:, 0, :]
    sample_count = values.shape[-1]
    spectrum = np.fft.rfft(values, axis=-1)
    amplitude = np.abs(spectrum) / sample_count
    if sample_count % 2 == 0:
        amplitude[:, 1:-1] *= 2.0
    else:
        amplitude[:, 1:] *= 2.0
    frequency_hz = np.fft.rfftfreq(sample_count, d=1.0 / fs)
    return frequency_hz, amplitude


def modal_metrics(modes, fs):
    frequency_hz, amplitude = single_sided_amplitude(modes, fs)
    energy = np.sum(np.asarray(modes, dtype=float)[:, 0, :] ** 2, axis=1)
    total = float(np.sum(energy))
    energy_fraction = (
        energy / total if total > np.finfo(float).eps
        else np.zeros_like(energy)
    )
    return {
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "energy": energy,
        "energy_fraction": energy_fraction,
    }


def add_modal_metrics(raw_result, fs):
    result = dict(raw_result)
    result.update(modal_metrics(raw_result["modes"], fs))
    return result
```

- [ ] **Step 7: Insert the analysis core and regenerate**

Add this cell after all three original source cells:

```python
code(ANALYSIS_CORE, "analysis-core", tags=("core",)),
```

Run:

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "adapters or single_sided or modal_metrics" -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the adapters and metrics**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "feat: add original VMD STVMD analysis adapters"
```

### Task 4: Add the six requested figures and optional export

**Files:**
- Modify: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Write failing figure and saving tests**

Append:

```python
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


def test_each_method_has_exactly_three_requested_figures(tmp_path):
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    figures = namespace["plot_method_results"](
        "VMD", time_s, fs, result, plot_max_hz=50.0
    )
    assert set(figures) == {"time_modes", "frequency_modes", "energy_fraction"}
    assert len(figures["time_modes"].axes) == 3
    assert len(figures["frequency_modes"].axes) == 3
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
    assert len(list(tmp_path.glob("*.png"))) == 6
    saved = np.load(
        tmp_path / "vmd_stvmd_single_waveform_results.npz"
    )
    assert saved["vmd_modes"].shape == (3, 1, 64)
    assert saved["stvmd_modes"].shape == (3, 1, 64)
    assert saved["vmd_energy_fraction"].sum() == pytest.approx(1.0)
    assert saved["stvmd_energy_fraction"].sum() == pytest.approx(1.0)
    for figure in (*vmd_figures.values(), *stvmd_figures.values()):
        matplotlib.pyplot.close(figure)
```

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "three_requested_figures or save_analysis" -q
```

Expected: failures because plotting and saving functions do not exist.

- [ ] **Step 3: Implement the three shared plotting functions**

Add:

```python
PLOTTING = r'''
def component_label(index):
    return "Residual" if index == 0 else f"Mode {index}"


def plot_modal_time(method, time_s, result):
    modes = result["modes"][:, 0, :]
    figure, axes = plt.subplots(
        modes.shape[0],
        1,
        figsize=(11, 2.0 * modes.shape[0]),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(time_s, modes[index], lw=0.75)
        axis.axvline(0.0, color="black", ls="--", lw=0.6)
        axis.set_ylabel(f"{component_label(index)}\n(mm/s)")
    axes[0].set_title(f"{method}: modal time histories")
    axes[-1].set_xlabel("Time (s)")
    return figure


def plot_modal_frequency(method, fs, result, plot_max_hz):
    amplitude = result["amplitude"]
    frequency_hz = result["frequency_hz"]
    figure, axes = plt.subplots(
        amplitude.shape[0],
        1,
        figsize=(11, 2.0 * amplitude.shape[0]),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(frequency_hz, amplitude[index], lw=0.8)
        axis.set_ylabel(f"{component_label(index)}\n(mm/s)")
        axis.set_xlim(0.0, min(float(plot_max_hz), fs / 2.0))
    axes[0].set_title(f"{method}: modal amplitude spectra")
    axes[-1].set_xlabel("Frequency (Hz)")
    return figure


def plot_energy_fraction(method, result):
    fractions = result["energy_fraction"]
    positions = np.arange(fractions.size)
    figure, axis = plt.subplots(
        1, 1, figsize=(10, 4.5), constrained_layout=True
    )
    bars = axis.bar(positions, fractions)
    axis.set_xticks(
        positions,
        [component_label(index) for index in positions],
    )
    axis.set(
        xlabel="Component",
        ylabel="Energy fraction",
        title=f"{method}: modal energy fractions",
    )
    axis.bar_label(bars, fmt="%.3f")
    return figure


def plot_method_results(method, time_s, fs, result, plot_max_hz):
    return {
        "time_modes": plot_modal_time(method, time_s, result),
        "frequency_modes": plot_modal_frequency(
            method, fs, result, plot_max_hz
        ),
        "energy_fraction": plot_energy_fraction(method, result),
    }
'''.strip()
```

- [ ] **Step 4: Implement six-PNG and NPZ saving**

Add:

```python
SAVING = r'''
def save_analysis(
    output_dir,
    waveform,
    vmd_result,
    stvmd_result,
    vmd_figures,
    stvmd_figures,
    parameters,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, figures in (
        ("vmd", vmd_figures),
        ("stvmd", stvmd_figures),
    ):
        for name, figure in figures.items():
            figure.savefig(
                output_dir / f"{method}_{name}.png",
                dpi=300,
                bbox_inches="tight",
            )
    np.savez_compressed(
        output_dir / "vmd_stvmd_single_waveform_results.npz",
        input_file=str(waveform.path),
        direction=waveform.direction,
        fs=waveform.fs,
        time_s=waveform.time_s,
        input_velocity=waveform.values,
        vmd_modes=vmd_result["modes"],
        vmd_frequency_hz=vmd_result["frequency_hz"],
        vmd_amplitude=vmd_result["amplitude"],
        vmd_energy=vmd_result["energy"],
        vmd_energy_fraction=vmd_result["energy_fraction"],
        vmd_center_frequency_hz=vmd_result["center_frequency_hz"],
        stvmd_modes=stvmd_result["modes"],
        stvmd_frequency_hz=stvmd_result["frequency_hz"],
        stvmd_amplitude=stvmd_result["amplitude"],
        stvmd_energy=stvmd_result["energy"],
        stvmd_energy_fraction=stvmd_result["energy_fraction"],
        stvmd_center_frequency_hz=stvmd_result["center_frequency_hz"],
        parameter_names=np.asarray(list(parameters), dtype=str),
        parameter_values=np.asarray(
            [str(value) for value in parameters.values()], dtype=str
        ),
    )
'''.strip()
```

- [ ] **Step 5: Insert plotting and saving cells and verify GREEN**

Insert after `ANALYSIS_CORE`:

```python
code(PLOTTING, "plotting", tags=("core",)),
code(SAVING, "saving", tags=("core",)),
```

Regenerate and run:

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "three_requested_figures or save_analysis" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit figures and output support**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "feat: add VMD STVMD modal figures"
```

### Task 5: Assemble, execute, and verify the complete notebook

**Files:**
- Modify: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Add the VMD-then-STVMD execution cells**

Add generated strings in this exact order:

```python
LOAD_AND_VALIDATE = r'''
waveform = load_single_waveform(INPUT_FILE, DIRECTION)
validate_config(
    waveform,
    K,
    ALPHA,
    TAU,
    TOL,
    MAX_ITERS,
    VMD_N_FFT,
    STVMD_WINDOW_LENGTH,
    PLOT_MAX_HZ,
)
vmd_memory_gb = estimate_vmd_memory_gb(
    1, waveform.values.size, K, VMD_N_FFT, MAX_ITERS
)
stvmd_memory_gb, stvmd_window_count = estimate_stvmd_memory_gb(
    1,
    waveform.values.size,
    K,
    STVMD_WINDOW_LENGTH,
    MAX_ITERS,
)
print("Input:", waveform.path.resolve())
print("Direction:", waveform.direction)
print("Sampling rate:", waveform.fs, "Hz")
print("Samples:", waveform.values.size)
print("STVMD hop length: 1")
print("STVMD window count:", stvmd_window_count)
print(f"Estimated VMD memory: {vmd_memory_gb:.2f} GB")
print(f"Estimated STVMD memory: {stvmd_memory_gb:.2f} GB")
if max(vmd_memory_gb, stvmd_memory_gb) > 4.0:
    warnings.warn(
        "Estimated solver memory exceeds 4 GB; reduce K, MAX_ITERS, "
        "or STVMD_WINDOW_LENGTH if necessary.",
        RuntimeWarning,
    )
'''.strip()


RUN_VMD = r'''
vmd_result = run_original_vmd(
    waveform.values.reshape(1, -1),
    fs=waveform.fs,
    K=K,
    alpha=ALPHA,
    tau=TAU,
    tol=TOL,
    max_iters=MAX_ITERS,
    n_fft=VMD_N_FFT,
)
vmd_result = add_modal_metrics(vmd_result, waveform.fs)
vmd_figures = plot_method_results(
    "VMD", waveform.time_s, waveform.fs, vmd_result, PLOT_MAX_HZ
)
for figure in vmd_figures.values():
    display(figure)
'''.strip()


RUN_STVMD = r'''
stvmd_result = run_original_stvmd(
    waveform.values.reshape(1, -1),
    fs=waveform.fs,
    K=K,
    alpha=ALPHA,
    tau=TAU,
    tol=TOL,
    max_iters=MAX_ITERS,
    window_length=STVMD_WINDOW_LENGTH,
)
stvmd_result = add_modal_metrics(stvmd_result, waveform.fs)
stvmd_figures = plot_method_results(
    "STVMD",
    waveform.time_s,
    waveform.fs,
    stvmd_result,
    PLOT_MAX_HZ,
)
for figure in stvmd_figures.values():
    display(figure)
'''.strip()


EXPORT = r'''
OUTPUT_DIR = Path("output/vmd_stvmd_single_waveform")
parameters = {
    "K": K,
    "ALPHA": ALPHA,
    "TAU": TAU,
    "TOL": TOL,
    "MAX_ITERS": MAX_ITERS,
    "VMD_N_FFT": VMD_N_FFT,
    "STVMD_WINDOW_LENGTH": STVMD_WINDOW_LENGTH,
    "HOP_LENGTH": 1,
    "PLOT_MAX_HZ": PLOT_MAX_HZ,
}
if SAVE_OUTPUTS:
    save_analysis(
        OUTPUT_DIR,
        waveform,
        vmd_result,
        stvmd_result,
        vmd_figures,
        stvmd_figures,
        parameters,
    )
    print("Saved to:", OUTPUT_DIR.resolve())
else:
    print("SAVE_OUTPUTS=False: no files were written")
'''.strip()
```

- [ ] **Step 2: Complete the notebook cell order**

Append after all core cells:

```python
markdown("## Load and validate one waveform", "load-heading"),
code(LOAD_AND_VALIDATE, "load-and-validate"),
markdown("## 1. Original VMD", "vmd-heading"),
code(RUN_VMD, "run-vmd"),
markdown("## 2. Original dynamic STVMD", "stvmd-heading"),
code(RUN_STVMD, "run-stvmd"),
markdown("## Save requested outputs", "export-heading"),
code(EXPORT, "export"),
```

- [ ] **Step 3: Add a patched end-to-end notebook test**

Append:

```python
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
PLOT_MAX_HZ = 64.0
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
```

- [ ] **Step 4: Regenerate and run the complete new test file**

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py -q
```

Expected: all tests in the new module pass.

- [ ] **Step 5: Verify deterministic generation**

```powershell
$before = (Get-FileHash -Algorithm SHA256 `
  single_waveform_vmd_stvmd_original.ipynb).Hash
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
$after = (Get-FileHash -Algorithm SHA256 `
  single_waveform_vmd_stvmd_original.ipynb).Hash
if ($before -ne $after) {
    throw "Notebook generation is not deterministic"
}
```

Expected: hashes are identical.

- [ ] **Step 6: Run all repository notebook tests**

Run in a clean feature worktree containing the tracked TXT files:

```powershell
python -m pytest -q `
  tests/test_blast_multichannel_stvmd_notebook.py `
  tests/test_single_waveform_stvmd_batched_notebook.py `
  tests/test_single_waveform_vmd_original_notebook.py `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
```

Expected: all old and new notebook tests pass.

- [ ] **Step 7: Commit final integration**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "test: verify original VMD STVMD notebook"
```

- [ ] **Step 8: Review the complete feature diff**

```powershell
git diff --check main..HEAD
git diff --stat main..HEAD
git status -sb
```

Expected: only the design/plan, new generator, new notebook, and new test file
belong to this feature; the feature worktree is clean.
