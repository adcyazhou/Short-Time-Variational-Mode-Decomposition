# Single-waveform Original VMD Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained notebook that reads one Instantel TXT direction and analyzes the complete waveform with the original repository VMD implementation.

**Architecture:** A deterministic generator extracts the buffer helpers and `VMD` class directly from `main_STVMD.ipynb` and embeds them unchanged. Notebook-specific adapter, diagnostics, plotting, and saving code surrounds the verbatim source without altering the algorithm.

**Tech Stack:** Python, Jupyter/nbformat/nbclient, NumPy, SciPy, Numba, Matplotlib, tqdm, pytest

---

### Task 1: Add failing notebook contract tests

**Files:**
- Create: `tests/test_single_waveform_vmd_original_notebook.py`
- Create later: `tools/build_single_waveform_vmd_original_notebook.py`
- Create later: `single_waveform_vmd_original.ipynb`

- [ ] **Step 1: Write artifact and parameter tests**

Create a test module with:

```python
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "single_waveform_vmd_original.ipynb"
GENERATOR = ROOT / "tools" / "build_single_waveform_vmd_original_notebook.py"
SOURCE_NOTEBOOK = ROOT / "main_STVMD.ipynb"


def joined_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_vmd_notebook_and_generator_exist():
    assert NOTEBOOK.is_file()
    assert GENERATOR.is_file()


def test_manual_parameters_have_one_source_of_truth():
    source = joined_source()
    for line in (
        'INPUT_FILE = Path("5m.TXT")',
        'DIRECTION = "Tran"',
        "K = 4",
        "ALPHA = 50.0",
        "N_FFT = 64",
        "TAU = 1e-5",
        "TOL = 1e-9",
        "MAX_ITERS = 10000",
        "PLOT_MAX_HZ = 200.0",
        "SAVE_OUTPUTS = True",
    ):
        assert source.count(line) == 1
    assert "STVMD_QUICK_TEST" not in source


def test_notebook_contains_original_vmd_pipeline():
    source = joined_source()
    for marker in (
        "class VMD(object):",
        "def load_single_waveform",
        "def run_original_vmd",
        "def summarize_vmd_result",
        "def plot_vmd_results",
        'OUTPUT_DIR = Path("output/vmd_single_waveform")',
    ):
        assert marker in source
```

- [ ] **Step 2: Run tests and verify the missing-artifact failure**

Run:

```powershell
python -m pytest tests/test_single_waveform_vmd_original_notebook.py -q
```

Expected: failures because the notebook and generator do not exist.

- [ ] **Step 3: Commit the failing tests**

Run:

```powershell
git add -- tests/test_single_waveform_vmd_original_notebook.py
git commit -m "test: define original VMD notebook contract"
```

### Task 2: Generate the original-VMD notebook

**Files:**
- Create: `tools/build_single_waveform_vmd_original_notebook.py`
- Generate: `single_waveform_vmd_original.ipynb`
- Test: `tests/test_single_waveform_vmd_original_notebook.py`

- [ ] **Step 1: Implement verbatim source extraction**

The generator reads `main_STVMD.ipynb` and uses:

```python
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


source_notebook = nbformat.read(ROOT / "main_STVMD.ipynb", as_version=4)
BUFFER_SOURCE = find_code_cell(
    source_notebook, ("def buffer(", "def unbuffer(", "def window_norm(")
)
VMD_SOURCE = find_code_cell(source_notebook, ("class VMD(object):",))
```

Embed `BUFFER_SOURCE` and `VMD_SOURCE` without trimming, reformatting, or
replacing text. Tag both generated cells `original-vmd-source`.

- [ ] **Step 2: Add imports and the manual parameter cell**

The generated notebook imports the names required by the original source:

```python
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython.display import display
from numba import jit, prange
from scipy.fft import irfft, rfft
from tqdm import tqdm
```

The sole parameter cell contains:

```python
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
N_FFT = 64
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 10000

PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True
```

- [ ] **Step 3: Add one-TXT loading and VMD adapter code**

Reuse the `InstantelRecord`, `SingleWaveform`, `load_instantel_txt`, and
`load_single_waveform` source from
`tools/build_single_waveform_stvmd_batched_notebook.py`.

Add validation and the adapter:

```python
def validate_vmd_config(K, alpha, n_fft, tau, tol, max_iters):
    if not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K 必须为不小于2的整数")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("N_FFT 必须为不小于2的整数")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS 必须为不小于2的整数")
    for name, value in (
        ("ALPHA", alpha),
        ("TAU", tau),
        ("TOL", tol),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")


def run_original_vmd(
    x, fs, K=4, alpha=50.0, n_fft=64,
    tau=1e-5, tol=1e-9, max_iters=10000,
):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("输入必须为有限的 (通道, 时间) 二维数组")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数")
    validate_vmd_config(K, alpha, n_fft, tau, tol, max_iters)
    model = VMD(
        num_channel=x.shape[0],
        n_fft=n_fft,
        alpha=alpha,
        K=K,
        tol=tol,
        tau=tau,
        maxiters=max_iters,
    )
    spectrum = model.prepare_offline(x)
    mode_spectrum, omega = model.apply(spectrum)
    modes = model.postprocess(mode_spectrum)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "center_freq_hz": omega * (fs / 2.0),
    }
```

- [ ] **Step 4: Add diagnostics, figures, and saving**

Implement `summarize_vmd_result(x, fs, raw)` to calculate:

```python
reconstruction = np.sum(raw["modes"], axis=0)
nrmse = np.linalg.norm(x - reconstruction, axis=1) / np.linalg.norm(x, axis=1)
mode_energy = np.sum(raw["modes"] ** 2, axis=2)
energy_fraction = mode_energy / np.sum(mode_energy, axis=0, keepdims=True)
```

Also calculate each component's full-record Fourier spectrum and 5%-95%
cumulative-energy frequency band.

`plot_vmd_results(waveform, result)` returns four figures:

```python
{
    "input_modes": ...,
    "center_frequencies": ...,
    "reconstruction_energy": ...,
    "mode_spectra": ...,
}
```

Label component zero `Residual`, show one horizontal/bar center frequency per
component, and do not draw time-varying frequency tracks.

`save_vmd_results` writes four PNG files and
`vmd_single_waveform_results.npz` to `output/vmd_single_waveform`.

- [ ] **Step 5: Generate the notebook and run contract tests**

Run:

```powershell
python tools/build_single_waveform_vmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_original_notebook.py -q
```

Expected: the notebook is valid JSON and the contract tests pass.

- [ ] **Step 6: Commit the generator and notebook**

Run:

```powershell
git add -- tools/build_single_waveform_vmd_original_notebook.py single_waveform_vmd_original.ipynb
git commit -m "feat: add single-waveform original VMD notebook"
```

### Task 3: Verify source identity and functionality

**Files:**
- Modify: `tests/test_single_waveform_vmd_original_notebook.py`
- Regenerate if needed: `single_waveform_vmd_original.ipynb`

- [ ] **Step 1: Test verbatim source identity**

Read both notebooks and assert:

```python
generated_sources = [
    "".join(cell.source)
    for cell in generated.cells
    if "original-vmd-source" in cell.metadata.get("tags", [])
]
source_buffer = find_source_cell(
    source, ("def buffer(", "def unbuffer(", "def window_norm(")
)
source_vmd = find_source_cell(source, ("class VMD(object):",))
assert generated_sources == [source_buffer, source_vmd]
```

- [ ] **Step 2: Test TXT selection and VMD result shapes**

Create a 64-row synthetic Instantel TXT and assert `Vert` selection. Run:

```python
raw = namespace["run_original_vmd"](
    signal.reshape(1, -1),
    fs=128,
    K=3,
    alpha=50.0,
    n_fft=16,
    tau=1e-5,
    tol=1e-4,
    max_iters=4,
)
assert raw["modes"].shape == (3, 1, 64)
assert raw["center_freq_hz"].shape == (3,)
```

- [ ] **Step 3: Test figures, output files, and a patched notebook execution**

Assert `plot_vmd_results` returns the four expected keys and
`save_vmd_results` writes four PNG files and one NPZ.

For the notebook execution test, patch the in-memory parameter cell to:

```python
INPUT_FILE = Path("<temporary synthetic TXT>")
DIRECTION = "Tran"
K = 3
ALPHA = 50.0
N_FFT = 16
TAU = 1e-5
TOL = 1e-4
MAX_ITERS = 4
PLOT_MAX_HZ = 64.0
SAVE_OUTPUTS = False
```

Execute with `NotebookClient(timeout=300, kernel_name="python3")` and verify
the committed notebook is unchanged.

- [ ] **Step 4: Run all focused notebook tests**

Run:

```powershell
python -m pytest tests/test_single_waveform_vmd_original_notebook.py tests/test_single_waveform_stvmd_batched_notebook.py tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify deterministic generation**

Run the generator between two SHA-256 calculations and require identical
notebook hashes.

- [ ] **Step 6: Commit functional tests**

Run:

```powershell
git add -- tests/test_single_waveform_vmd_original_notebook.py single_waveform_vmd_original.ipynb
git commit -m "test: verify single-waveform original VMD notebook"
```
