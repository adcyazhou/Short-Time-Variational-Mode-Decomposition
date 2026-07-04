# Single-waveform Batched STVMD Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained Jupyter notebook that reads one Instantel TXT waveform, selects one direction, runs batch-optimized dynamic STVMD with manually editable parameters, and produces diagnostic figures and saved results.

**Architecture:** A deterministic generator builds the notebook using the existing batch-optimized STVMD source as the algorithm baseline while adapting only its window validation for arbitrary integer lengths. A dedicated test module checks the configuration contract, TXT parsing, analysis shapes, plotting, saving, and a small notebook execution created by patching a notebook copy in memory rather than exposing a user-facing quick-test switch.

**Tech Stack:** Python, Jupyter/nbformat/nbclient, NumPy, SciPy, Numba, pandas, Matplotlib, tqdm, pytest

---

### Task 1: Define the single-waveform notebook contract with failing tests

**Files:**
- Create: `tests/test_single_waveform_stvmd_batched_notebook.py`
- Create later: `single_waveform_stvmd_batched.ipynb`
- Create later: `tools/build_single_waveform_stvmd_batched_notebook.py`

- [ ] **Step 1: Add structure and parameter tests**

Create `tests/test_single_waveform_stvmd_batched_notebook.py` with:

```python
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "single_waveform_stvmd_batched.ipynb"
GENERATOR = ROOT / "tools" / "build_single_waveform_stvmd_batched_notebook.py"


def joined_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def notebook_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "core" in cell.metadata.get("tags", []):
            exec("".join(cell.source), namespace)
    return namespace


def test_single_waveform_notebook_and_generator_exist():
    assert NOTEBOOK.is_file()
    assert GENERATOR.is_file()


def test_manual_parameter_cell_has_one_source_of_truth():
    source = joined_source()
    for line in (
        'INPUT_FILE = Path("5m.TXT")',
        'DIRECTION = "Tran"',
        "K = 4",
        "ALPHA = 50.0",
        "WINDOW_LENGTH = 64",
        "TAU = 1e-5",
        "TOL = 1e-5",
        "MAX_ITERS = 300",
        "BATCH_WINDOWS = 128",
        "PLOT_MAX_HZ = 200.0",
        "SAVE_OUTPUTS = True",
    ):
        assert line in source
    assert "STVMD_QUICK_TEST" not in source
    assert "REPOSITORY_WINDOWS" not in source


def test_notebook_contains_single_waveform_pipeline():
    source = joined_source()
    for marker in (
        "def load_single_waveform",
        "def run_dynamic_stvmd_batched",
        "def analyze_single_waveform",
        "def plot_single_waveform_results",
        'OUTPUT_DIR = Path("output/stvmd_single_waveform")',
    ):
        assert marker in source
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_single_waveform_stvmd_batched_notebook.py -q
```

Expected: failures because the generator and notebook do not exist.

- [ ] **Step 3: Commit the failing contract tests**

Run:

```powershell
git add -- tests/test_single_waveform_stvmd_batched_notebook.py
git commit -m "test: define single-waveform STVMD notebook contract"
```

Expected: one commit containing only the new test file.

### Task 2: Build the manual-parameter notebook and analysis pipeline

**Files:**
- Create: `tools/build_single_waveform_stvmd_batched_notebook.py`
- Generate: `single_waveform_stvmd_batched.ipynb`
- Test: `tests/test_single_waveform_stvmd_batched_notebook.py`

- [ ] **Step 1: Add the deterministic notebook generator**

Create `tools/build_single_waveform_stvmd_batched_notebook.py` with these units:

```python
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from build_blast_multichannel_stvmd_notebook import (
    DIAGNOSTICS,
    STVMD as MULTICHANNEL_BATCHED_STVMD,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "single_waveform_stvmd_batched.ipynb"


def single_stvmd_source():
    source = MULTICHANNEL_BATCHED_STVMD
    source = source.replace(
        "REPOSITORY_WINDOWS = (8, 16, 32, 64, 128, 256)\\n\\n\\n",
        "",
    )
    old = (
        "    if window_length not in REPOSITORY_WINDOWS:\\n"
        "        raise ValueError(f\"WINDOW_LENGTH 必须来自 {REPOSITORY_WINDOWS}\")\\n"
    )
    new = (
        "    if not isinstance(window_length, (int, np.integer)) "
        "or window_length < 2:\\n"
        "        raise ValueError(\"WINDOW_LENGTH 必须为不小于2的整数\")\\n"
    )
    if source.count(old) != 1:
        raise RuntimeError("Unexpected shared STVMD validation source")
    return source.replace(old, new)
```

The generator must define a manual configuration cell without an environment
override:

```python
INPUT_FILE = Path("5m.TXT")
DIRECTION = "Tran"

K = 4
ALPHA = 50.0
WINDOW_LENGTH = 64
TAU = 1e-5
TOL = 1e-5
MAX_ITERS = 300
BATCH_WINDOWS = 128
PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = True
```

- [ ] **Step 2: Add the one-file Instantel loader**

Embed this public interface in the generated notebook:

```python
@dataclass(frozen=True)
class SingleWaveform:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    direction: str
    time_s: np.ndarray
    values: np.ndarray


def load_single_waveform(path, direction):
    direction_names = ("Tran", "Vert", "Long")
    if direction not in direction_names:
        raise ValueError(f"DIRECTION 必须为 {direction_names} 之一")
    record = load_instantel_txt(path)
    column = direction_names.index(direction)
    values = record.data[:, column].astype(float, copy=False)
    time_s = np.arange(values.size) / record.fs - record.pretrigger_seconds
    return SingleWaveform(
        path=record.path,
        metadata=record.metadata,
        fs=record.fs,
        pretrigger_seconds=record.pretrigger_seconds,
        direction=direction,
        time_s=time_s,
        values=values,
    )
```

The embedded `load_instantel_txt` must parse metadata and the three-column
numeric table exactly as the current blast notebook does, but it must not
create CSV files or load multiple records.

- [ ] **Step 3: Add single-waveform analysis, plotting, and saving**

Embed these interfaces:

```python
def analyze_single_waveform(waveform):
    x = waveform.values.reshape(1, -1)
    raw = run_dynamic_stvmd_batched(
        x,
        fs=waveform.fs,
        K=K,
        alpha=ALPHA,
        window_length=WINDOW_LENGTH,
        tau=TAU,
        tol=TOL,
        max_iters=MAX_ITERS,
        batch_windows=BATCH_WINDOWS,
    )
    return summarize_stvmd_result(x, waveform.fs, raw)


def plot_single_waveform_results(waveform, result):
    return {
        "input_tf": plot_input_and_tf(waveform, result, PLOT_MAX_HZ),
        "modes": plot_modes(waveform, result),
        "if_reconstruction": plot_if_and_reconstruction(
            waveform, result, PLOT_MAX_HZ
        ),
        "spectrum_if_mapping": plot_spectrum_if_mapping(
            waveform, result, PLOT_MAX_HZ
        ),
    }


def save_single_waveform_results(output_dir, waveform, result, figures):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(
            output_dir / f"{waveform.direction.lower()}_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
    np.savez_compressed(
        output_dir / "stvmd_single_waveform_results.npz",
        direction=waveform.direction,
        fs=waveform.fs,
        time_s=waveform.time_s,
        modes=result["modes"],
        center_freq_hz=result["center_freq_hz"],
        reconstruction=result["reconstruction"],
        nrmse=result["nrmse"],
        energy_fraction=result["energy_fraction"],
        frequency_bands_hz=result["frequency_bands_hz"],
    )
```

The four plotting functions must use the selected waveform only, label
component zero `Residual`, plot center-frequency tracks only for components
`1..K-1`, and retain the paper-style spectrum/IF mapping.

- [ ] **Step 4: Build the notebook**

Run:

```powershell
python tools/build_single_waveform_stvmd_batched_notebook.py
```

Expected: `single_waveform_stvmd_batched.ipynb` is created as valid notebook
JSON with deterministic cell IDs and no execution outputs.

- [ ] **Step 5: Run the structure tests**

Run:

```powershell
python -m pytest tests/test_single_waveform_stvmd_batched_notebook.py -q
```

Expected: the three contract tests pass.

- [ ] **Step 6: Commit the generator and notebook**

Run:

```powershell
git add -- tools/build_single_waveform_stvmd_batched_notebook.py single_waveform_stvmd_batched.ipynb
git commit -m "feat: add single-waveform batched STVMD notebook"
```

Expected: the generator and generated notebook are committed together.

### Task 3: Add functional and notebook smoke tests

**Files:**
- Modify: `tests/test_single_waveform_stvmd_batched_notebook.py`
- Regenerate if needed: `single_waveform_stvmd_batched.ipynb`

- [ ] **Step 1: Add parser and algorithm shape tests**

Append tests that create a small Instantel file:

```python
def instantel_text(rows, fs=128, pretrigger=0.5):
    body = "\n".join(f"{a} {b} {c}" for a, b, c in rows)
    return (
        f"Sample Rate: {fs} Hz\n"
        f"Pre-trigger Length: {pretrigger} sec\n"
        "Tran Vert Long\n"
        f"{body}\n"
    )


def test_load_single_waveform_selects_requested_direction(tmp_path):
    path = tmp_path / "single.TXT"
    path.write_text(
        instantel_text([(1, 10, 100), (2, 20, 200)]),
        encoding="utf-8",
    )
    namespace = notebook_namespace()
    waveform = namespace["load_single_waveform"](path, "Vert")
    np.testing.assert_array_equal(waveform.values, [10.0, 20.0])
    assert waveform.fs == 128
    assert waveform.time_s[0] == -0.5


def test_batched_analysis_returns_single_channel_shapes():
    namespace = notebook_namespace()
    t = np.arange(64) / 128
    x = np.sin(2 * np.pi * 15 * t).reshape(1, -1)
    result = namespace["run_dynamic_stvmd_batched"](
        x,
        fs=128,
        K=3,
        alpha=50.0,
        window_length=16,
        tau=1e-5,
        tol=1e-4,
        max_iters=4,
        batch_windows=8,
    )
    assert result["modes"].shape == (3, 1, 64)
    assert result["center_freq_hz"].shape == (3, 64)
```

- [ ] **Step 2: Add plotting and save tests**

Use the same 64-point synthetic waveform to assert:

```python
figures = namespace["plot_single_waveform_results"](waveform, summary)
assert set(figures) == {
    "input_tf",
    "modes",
    "if_reconstruction",
    "spectrum_if_mapping",
}
namespace["save_single_waveform_results"](tmp_path, waveform, summary, figures)
assert len(list(tmp_path.glob("*.png"))) == 4
assert (tmp_path / "stvmd_single_waveform_results.npz").is_file()
```

- [ ] **Step 3: Add a complete patched-copy notebook execution test**

Read the notebook with `nbformat`, replace only the manual configuration cell
in the in-memory copy with a temporary input path and:

```python
K = 3
ALPHA = 50.0
WINDOW_LENGTH = 8
TAU = 1e-5
TOL = 1e-4
MAX_ITERS = 3
BATCH_WINDOWS = 4
PLOT_MAX_HZ = 64.0
SAVE_OUTPUTS = False
```

Execute that copy with `NotebookClient(timeout=300, kernel_name="python3")`
and assert that execution completes without modifying the committed notebook.

- [ ] **Step 4: Run the new and existing focused suites**

Run:

```powershell
python -m pytest tests/test_single_waveform_stvmd_batched_notebook.py tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify deterministic generation**

Run:

```powershell
$before = (Get-FileHash -Algorithm SHA256 single_waveform_stvmd_batched.ipynb).Hash
python tools/build_single_waveform_stvmd_batched_notebook.py
$after = (Get-FileHash -Algorithm SHA256 single_waveform_stvmd_batched.ipynb).Hash
if ($before -ne $after) { throw "Notebook generation is not deterministic" }
Write-Output "DETERMINISTIC=True"
```

Expected: `DETERMINISTIC=True`.

- [ ] **Step 6: Commit the functional tests**

Run:

```powershell
git add -- tests/test_single_waveform_stvmd_batched_notebook.py single_waveform_stvmd_batched.ipynb
git commit -m "test: verify single-waveform STVMD notebook"
```

Expected: the test additions and any deterministic notebook regeneration are
committed without staging TXT data files.
