# Blast Individual VMD with Custom Centers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Notebook that independently decomposes the Tran, Vert, and Long channels in 5m/10m/15m Instantel records with user-specified K and warm-start center frequencies, fixed `alpha=2000`, and one original-plus-modes figure per signal.

**Architecture:** A deterministic Python builder generates the Notebook from focused source strings. The Notebook contains an Instantel reader, strict per-signal configuration validation, and a two-buffer implementation of the source Notebook's VMD ADMM equations whose user centers initialize but do not freeze the center updates. Tests execute tagged Notebook cells directly and use short synthetic signals plus a reduced quick path, while the delivered Notebook retains full-record defaults.

**Tech Stack:** Python 3, Jupyter/nbformat, NumPy, SciPy FFT, Matplotlib, pytest.

---

## File structure

- Create `tools/build_blast_individual_vmd_custom_centers_notebook.py`: deterministic builder and all self-contained Notebook cell sources.
- Create `tests/test_blast_individual_vmd_custom_centers_notebook.py`: builder contract, parsing, validation, VMD behavior, plotting, and quick integration tests.
- Generate `blast_individual_vmd_custom_centers.ipynb`: user-facing analysis artifact; no runtime dependency on the builder.
- Modify `docs/superpowers/specs/2026-07-13-blast-individual-vmd-custom-centers-design.md`: already corrected to document the source VMD normalized-frequency conversion and two-buffer storage.

### Task 1: Scaffold the deterministic Notebook contract

**Files:**
- Create: `tests/test_blast_individual_vmd_custom_centers_notebook.py`
- Create: `tools/build_blast_individual_vmd_custom_centers_notebook.py`
- Generate: `blast_individual_vmd_custom_centers.ipynb`

- [ ] **Step 1: Write the failing builder-contract test**

Create the test constants and first test:

```python
from pathlib import Path
import os
import subprocess
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_blast_individual_vmd_custom_centers_notebook.py"
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
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py::test_builder_generates_ordered_self_contained_notebook -v
```

Expected: FAIL because the builder and target Notebook do not exist.

- [ ] **Step 3: Implement the minimal deterministic builder**

Create the builder with the exact cell helper and initial cells:

```python
from pathlib import Path

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import nbformat


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "blast_individual_vmd_custom_centers.ipynb"


def markdown(source, cell_id):
    return new_markdown_cell(source, id=cell_id)


def code(source, cell_id):
    return new_code_cell(source, id=cell_id)


TITLE = """# 5m/10m/15m 九通道自定义中心频率 VMD\n\n算法更新方程来自 `figure_experiment_STVMD_ssvep_singlechannel.ipynb`。用户中心仅作为初值，后续中心继续迭代；alpha 固定为 2000。"""

IMPORTS = """from dataclasses import dataclass\nfrom pathlib import Path\nimport os\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom IPython.display import display\nfrom scipy.fft import irfft, rfft"""

CONFIG = "ALPHA = 2000.0\nVMD_CONFIG = {}"
PLACEHOLDER = "pass"


def build():
    notebook = new_notebook(cells=[
        markdown(TITLE, "title"),
        code(IMPORTS, "imports"),
        code(CONFIG, "config"),
        code(PLACEHOLDER, "loader"),
        code(PLACEHOLDER, "validation"),
        code(PLACEHOLDER, "warm-start-vmd"),
        code(PLACEHOLDER, "analysis"),
        code(PLACEHOLDER, "plotting"),
        code(PLACEHOLDER, "load-records"),
        code(PLACEHOLDER, "run-all"),
    ])
    nbformat.write(notebook, TARGET)


if __name__ == "__main__":
    build()
```

- [ ] **Step 4: Generate the Notebook and verify GREEN**

Run:

```powershell
python tools/build_blast_individual_vmd_custom_centers_notebook.py
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py::test_builder_generates_ordered_self_contained_notebook -v
```

Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

```powershell
git add tools/build_blast_individual_vmd_custom_centers_notebook.py tests/test_blast_individual_vmd_custom_centers_notebook.py blast_individual_vmd_custom_centers.ipynb
git commit -m "feat: scaffold custom-center blast VMD notebook"
```

### Task 2: Add nine-signal configuration and Instantel parsing

**Files:**
- Modify: `tests/test_blast_individual_vmd_custom_centers_notebook.py`
- Modify: `tools/build_blast_individual_vmd_custom_centers_notebook.py`
- Regenerate: `blast_individual_vmd_custom_centers.ipynb`

- [ ] **Step 1: Add helpers for executing tagged Notebook cells in tests**

```python
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
```

- [ ] **Step 2: Write failing parser and configuration-shape tests**

```python
def test_config_contains_independent_entries_for_all_nine_signals():
    ns = notebook_namespace("imports", "config")
    assert ns["ALPHA"] == 2000.0
    assert set(ns["VMD_CONFIG"]) == {"5m", "10m", "15m"}
    for distance in ("5m", "10m", "15m"):
        assert set(ns["VMD_CONFIG"][distance]) == {"Tran", "Vert", "Long"}
        for config in ns["VMD_CONFIG"][distance].values():
            assert config["K"] == len(config["centers_hz"])


def test_load_instantel_record_preserves_all_three_channels(tmp_path):
    path = tmp_path / "record.TXT"
    write_instantel(path, fs=256)
    ns = notebook_namespace("imports", "loader")
    record = ns["load_instantel_record"](path)
    assert record.fs == 256
    assert record.pretrigger_seconds == 0.5
    np.testing.assert_allclose(record.channels["Tran"], [1, 4])
    np.testing.assert_allclose(record.channels["Vert"], [2, 5])
    np.testing.assert_allclose(record.channels["Long"], [3, 6])
    np.testing.assert_allclose(record.time_s, [-0.5, -0.5 + 1 / 256])
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "config_contains or load_instantel" -v
```

Expected: FAIL because the config is empty and the loader is absent.

- [ ] **Step 4: Implement the complete config cell**

Replace `CONFIG` with:

```python
CONFIG = r'''
INPUT_FILES = {
    "5m": Path("5m.TXT"),
    "10m": Path("10m.TXT"),
    "15m": Path("15m.TXT"),
}

# Edit every K and centers_hz list independently.
# centers_hz contains all K initial centers; 0 Hz is never added automatically.
VMD_CONFIG = {
    distance: {
        direction: {"K": 3, "centers_hz": [10.0, 40.0, 100.0]}
        for direction in ("Tran", "Vert", "Long")
    }
    for distance in ("5m", "10m", "15m")
}

ALPHA = 2000.0
N_FFT = 64
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 10000
PLOT_DPI = 120

QUICK_TEST = os.environ.get("BLAST_VMD_QUICK_TEST") == "1"
if QUICK_TEST:
    MAX_ITERS = 20
'''.strip()
```

- [ ] **Step 5: Implement the Instantel reader**

Use a focused dataclass and reject malformed data:

```python
LOADER = r'''
@dataclass(frozen=True)
class BlastRecord:
    path: Path
    fs: float
    pretrigger_seconds: float
    unit: str
    channels: dict
    time_s: np.ndarray


def _metadata_number(metadata, key):
    if key not in metadata:
        raise ValueError(f"missing metadata: {key}")
    token = metadata[key].split()[0]
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"invalid {key}: {metadata[key]}") from exc


def load_instantel_record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    metadata, header_index = {}, None
    for index, raw in enumerate(lines):
        stripped = raw.strip().strip('"')
        if all(name in stripped for name in ("Tran", "Vert", "Long")):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()
    if header_index is None:
        raise ValueError(f"{path.name}: missing Tran/Vert/Long header")
    data = np.atleast_2d(np.loadtxt(lines[header_index + 1:], dtype=float))
    if data.shape[1] != 3 or data.shape[0] == 0 or not np.isfinite(data).all():
        raise ValueError(f"{path.name}: expected finite three-column data")
    fs = _metadata_number(metadata, "Sample Rate")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{path.name}: sample rate must be positive")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    time_s = np.arange(data.shape[0], dtype=float) / fs - pretrigger
    channels = {
        name: data[:, index].copy()
        for index, name in enumerate(("Tran", "Vert", "Long"))
    }
    return BlastRecord(
        path=path,
        fs=fs,
        pretrigger_seconds=pretrigger,
        unit="mm/s",
        channels=channels,
        time_s=time_s,
    )
'''.strip()
```

- [ ] **Step 6: Regenerate and verify GREEN**

```powershell
python tools/build_blast_individual_vmd_custom_centers_notebook.py
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "config_contains or load_instantel" -v
```

Expected: both tests PASS.

- [ ] **Step 7: Commit parser and configuration**

```powershell
git add tools/build_blast_individual_vmd_custom_centers_notebook.py tests/test_blast_individual_vmd_custom_centers_notebook.py blast_individual_vmd_custom_centers.ipynb
git commit -m "feat: load and configure nine blast signals"
```

### Task 3: Implement strict validation and warm-start VMD

**Files:**
- Modify: `tests/test_blast_individual_vmd_custom_centers_notebook.py`
- Modify: `tools/build_blast_individual_vmd_custom_centers_notebook.py`
- Regenerate: `blast_individual_vmd_custom_centers.ipynb`

- [ ] **Step 1: Write failing validation tests**

```python
import pytest


@pytest.mark.parametrize(
    "config, message",
    [
        ({"K": 3, "centers_hz": [10, 20]}, "K=3"),
        ({"K": 3, "centers_hz": [10, 10, 20]}, "strictly increasing"),
        ({"K": 3, "centers_hz": [20, 10, 30]}, "strictly increasing"),
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
        "5m", "Tran", {"K": 2, "centers_hz": [10, 20]}, fs=128
    )
    np.testing.assert_allclose(centers, [10, 20])
```

- [ ] **Step 2: Write failing warm-start behavior tests**

```python
def test_warm_start_vmd_updates_centers_and_reconstructs_two_sines():
    ns = notebook_namespace("imports", "validation", "warm-start-vmd", "analysis")
    fs = 128.0
    time_s = np.arange(512) / fs
    signal = np.sin(2 * np.pi * 20 * time_s) + 0.6 * np.sin(2 * np.pi * 28 * time_s)
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
    assert result["reconstruction_rmse"] < 0.1
    assert result["iterations"] <= 100


def test_hz_normalization_maps_nyquist_to_one():
    ns = notebook_namespace("imports", "validation", "warm-start-vmd")
    np.testing.assert_allclose(ns["centers_hz_to_internal"]([0, 32], 128), [0, 0.5])
    np.testing.assert_allclose(ns["centers_internal_to_hz"]([0, 0.5], 128), [0, 32])
```

- [ ] **Step 3: Run the tests and verify RED**

```powershell
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "validate_signal_config or warm_start_vmd or hz_normalization" -v
```

Expected: FAIL because validation and warm-start VMD functions are absent.

- [ ] **Step 4: Implement strict validation**

```python
VALIDATION = r'''
def validate_signal_config(distance, direction, config, fs):
    key = f"{distance}/{direction}"
    K = config.get("K")
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(f"{key}: K must be a positive integer")
    centers = np.asarray(config.get("centers_hz"), dtype=float)
    if centers.ndim != 1 or centers.size != K:
        raise ValueError(f"{key}: K={K} but centers_hz has {centers.size} values")
    if not np.isfinite(centers).all():
        raise ValueError(f"{key}: centers_hz must be finite")
    if np.any(np.diff(centers) <= 0):
        raise ValueError(f"{key}: centers_hz must be strictly increasing")
    if np.any(centers < 0) or np.any(centers >= fs / 2.0):
        raise ValueError(f"{key}: centers_hz must lie below the Nyquist frequency")
    return centers


def validate_global_config(alpha, n_fft, tau, tol, max_iters):
    if alpha != 2000.0:
        raise ValueError("ALPHA is fixed at 2000.0")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("N_FFT must be an integer >= 2")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("TAU must be finite and nonnegative")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("TOL must be finite and positive")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 2:
        raise ValueError("MAX_ITERS must be an integer >= 2")
'''.strip()
```

- [ ] **Step 5: Implement the two-buffer source-equation VMD**

Implement `WarmStartVMD` in the `warm-start-vmd` cell. Preserve the source reflection padding, `rfft`, per-mode Wiener update, dual ascent, and `irfft`. The critical initialization and update must be:

```python
def centers_hz_to_internal(centers_hz, fs):
    return np.asarray(centers_hz, dtype=float) / (fs / 2.0)


def centers_internal_to_hz(centers, fs):
    return np.asarray(centers, dtype=float) * (fs / 2.0)


class WarmStartVMD:
    def __init__(self, num_channel, n_fft=64, alpha=2000.0, K=3,
                 tol=1e-9, tau=1e-5, maxiters=10000):
        self.num_channel = num_channel
        self.n_fft = n_fft
        self.alpha = alpha * np.ones(K)
        self.K = K
        self.tol = tol
        self.tau = tau
        self.maxiters = maxiters
        self.padwidth = (
            ((n_fft - 1) // 2, (n_fft - 1) // 2)
            if (n_fft - 1) % 2 == 0
            else ((n_fft - 1) // 2 + 1, (n_fft - 1) // 2)
        )

    def prepare_offline(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[0] != self.num_channel:
            raise ValueError("x must have shape (num_channel, samples)")
        self.len_x = x.shape[1]
        padded = np.pad(x, ((0, 0), self.padwidth), mode="reflect")
        return rfft(padded, axis=1, workers=-1)

    def apply(self, f_hat_plus, omega_init):
        channels, frequency_bins = f_hat_plus.shape
        freqs = np.arange(1, frequency_bins + 1, dtype=float) / frequency_bins
        omega_state = np.zeros((2, self.K), dtype=float)
        omega_state[0] = np.asarray(omega_init, dtype=float)
        u_state = np.zeros((2, channels, frequency_bins, self.K), complex)
        lambda_state = np.zeros((2, channels, frequency_bins), complex)
        sum_uk = np.zeros((channels, frequency_bins), complex)
        converged = False

        for iteration in range(self.maxiters - 1):
            current, next_index = iteration % 2, (iteration + 1) % 2
            for k in range(self.K):
                previous_k = self.K - 1 if k == 0 else k - 1
                previous_state = current if k == 0 else next_index
                sum_uk = (
                    u_state[previous_state, :, :, previous_k]
                    + sum_uk
                    - u_state[current, :, :, k]
                )
                denominator = 1.0 + self.alpha[k] * (
                    freqs - omega_state[current, k]
                ) ** 2
                u_state[next_index, :, :, k] = (
                    f_hat_plus - sum_uk - lambda_state[current] / 2.0
                ) / denominator
                power = np.abs(u_state[next_index, :, :, k]) ** 2
                total_power = float(np.sum(power))
                if not np.isfinite(total_power) or total_power <= np.finfo(float).eps:
                    raise FloatingPointError(f"mode {k + 1} has zero or invalid energy")
                omega_state[next_index, k] = np.sum(
                    freqs * np.sum(power, axis=0)
                ) / total_power

            lambda_state[next_index] = lambda_state[current] + self.tau * (
                np.sum(u_state[next_index], axis=2) - f_hat_plus
            )
            delta = u_state[next_index] - u_state[current]
            u_diff = float(np.mean(np.abs(delta) ** 2))
            if u_diff < self.tol and iteration > 2:
                converged = True
                break

        final_index = (iteration + 1) % 2
        order = np.argsort(omega_state[final_index])
        return (
            u_state[final_index, :, :, order],
            omega_state[final_index, order],
            iteration + 1,
            converged,
            order,
        )

    def postprocess(self, u_hat):
        padded_n = self.len_x + sum(self.padwidth)
        u = irfft(u_hat, n=padded_n, axis=1, workers=-1).real
        u = np.transpose(u, (2, 0, 1))
        return u[:, :, self.padwidth[0]:padded_n - self.padwidth[1]]
```

- [ ] **Step 6: Implement the analysis adapter**

```python
def run_warm_start_vmd(signal, fs, K, centers_hz, alpha, n_fft,
                       tau, tol, max_iters, data_key):
    distance, direction = data_key
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError(f"{distance}/{direction}: signal must be finite and one-dimensional")
    validate_global_config(alpha, n_fft, tau, tol, max_iters)
    centers = validate_signal_config(
        distance, direction, {"K": K, "centers_hz": centers_hz}, fs
    )
    model = WarmStartVMD(1, n_fft, alpha, K, tol, tau, max_iters)
    spectrum = model.prepare_offline(signal.reshape(1, -1))
    mode_spectrum, omega, iterations, converged, order = model.apply(
        spectrum, centers_hz_to_internal(centers, fs)
    )
    modes = model.postprocess(mode_spectrum)[:, 0, :]
    reconstruction = np.sum(modes, axis=0)
    return {
        "modes": modes,
        "mode_spectrum": mode_spectrum,
        "initial_centers_hz": centers[order],
        "final_centers_hz": centers_internal_to_hz(omega, fs),
        "reconstruction": reconstruction,
        "reconstruction_rmse": float(np.sqrt(np.mean((signal - reconstruction) ** 2))),
        "iterations": iterations,
        "converged": converged,
    }
```

- [ ] **Step 7: Regenerate and verify GREEN**

```powershell
python tools/build_blast_individual_vmd_custom_centers_notebook.py
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "validate_signal_config or warm_start_vmd or hz_normalization" -v
```

Expected: all selected tests PASS. If the synthetic RMSE threshold exposes the source algorithm's dual-ascent sign or tolerance behavior, diagnose using the source equations; do not weaken the assertion merely to obtain green.

- [ ] **Step 8: Commit the VMD implementation**

```powershell
git add tools/build_blast_individual_vmd_custom_centers_notebook.py tests/test_blast_individual_vmd_custom_centers_notebook.py blast_individual_vmd_custom_centers.ipynb
git commit -m "feat: add warm-start center VMD solver"
```

### Task 4: Add per-signal figures and the nine-analysis loop

**Files:**
- Modify: `tests/test_blast_individual_vmd_custom_centers_notebook.py`
- Modify: `tools/build_blast_individual_vmd_custom_centers_notebook.py`
- Regenerate: `blast_individual_vmd_custom_centers.ipynb`

- [ ] **Step 1: Write failing plotting tests**

```python
def test_plot_vmd_modes_creates_original_plus_one_axis_per_mode():
    ns = notebook_namespace("imports", "plotting")
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
    assert labels == ["Original\n(mm/s)", "Mode 1\n(mm/s)", "Mode 2\n(mm/s)"]
    assert "init=10.00 Hz, final=11.00 Hz" in figure.axes[1].get_title()
    assert "alpha=2000" in figure._suptitle.get_text()
    plt.close(figure)
```

- [ ] **Step 2: Write failing nine-analysis orchestration test**

```python
def test_analyze_all_records_runs_every_distance_direction_pair():
    ns = notebook_namespace("imports", "loader", "validation", "analysis")
    calls = []

    def fake_run(signal, **kwargs):
        calls.append(kwargs["data_key"])
        return {"modes": np.zeros((kwargs["K"], len(signal)))}

    ns["run_warm_start_vmd"] = fake_run
    record = ns["BlastRecord"](
        path=Path("x"), fs=128.0, pretrigger_seconds=0.5, unit="mm/s",
        channels={name: np.ones(16) for name in ("Tran", "Vert", "Long")},
        time_s=np.arange(16) / 128 - 0.5,
    )
    records = {distance: record for distance in ("5m", "10m", "15m")}
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
```

- [ ] **Step 3: Run the tests and verify RED**

```powershell
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "plot_vmd_modes or analyze_all_records" -v
```

Expected: FAIL because plotting and orchestration functions are absent.

- [ ] **Step 4: Implement the plotting function**

```python
def plot_vmd_modes(distance, direction, time_s, signal, result, alpha=2000.0):
    modes = result["modes"]
    figure, axes = plt.subplots(
        modes.shape[0] + 1, 1,
        figsize=(12, 2.0 * (modes.shape[0] + 1)),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(time_s, signal, color="#202020", linewidth=0.8)
    axes[0].set_ylabel("Original\n(mm/s)")
    axes[0].grid(alpha=0.2)
    for index, mode in enumerate(modes):
        axis = axes[index + 1]
        axis.plot(time_s, mode, color="#0072B2", linewidth=0.75)
        axis.set_ylabel(f"Mode {index + 1}\n(mm/s)")
        axis.set_title(
            f"init={result['initial_centers_hz'][index]:.2f} Hz, "
            f"final={result['final_centers_hz'][index]:.2f} Hz",
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Relative time (s)")
    figure.suptitle(
        f"{distance} {direction}: K={modes.shape[0]}, alpha={alpha:g}"
    )
    return figure
```

- [ ] **Step 5: Implement ordered orchestration and summaries**

```python
def analyze_all_records(records, config, alpha, n_fft, tau, tol, max_iters):
    results = {}
    for distance in ("5m", "10m", "15m"):
        record = records[distance]
        for direction in ("Tran", "Vert", "Long"):
            item = config[distance][direction]
            result = run_warm_start_vmd(
                record.channels[direction], fs=record.fs,
                K=item["K"], centers_hz=item["centers_hz"],
                alpha=alpha, n_fft=n_fft, tau=tau, tol=tol,
                max_iters=max_iters, data_key=(distance, direction),
            )
            results[(distance, direction)] = result
    return results


def print_vmd_summary(distance, direction, record, result):
    print(f"{distance}/{direction}: samples={record.time_s.size}, fs={record.fs:g} Hz")
    print("  initial centers (Hz):", np.array2string(result["initial_centers_hz"], precision=3))
    print("  final centers (Hz):  ", np.array2string(result["final_centers_hz"], precision=3))
    print(f"  iterations={result['iterations']}, converged={result['converged']}")
    print(f"  reconstruction RMSE={result['reconstruction_rmse']:.6g}")
```

- [ ] **Step 6: Fill the final two Notebook execution cells**

`load-records`:

```python
records = {
    distance: load_instantel_record(path)
    for distance, path in INPUT_FILES.items()
}
```

`run-all`:

```python
results = analyze_all_records(
    records, VMD_CONFIG, ALPHA, N_FFT, TAU, TOL, MAX_ITERS
)
figures = {}
for distance in ("5m", "10m", "15m"):
    for direction in ("Tran", "Vert", "Long"):
        key = (distance, direction)
        print_vmd_summary(distance, direction, records[distance], results[key])
        figures[key] = plot_vmd_modes(
            distance, direction,
            records[distance].time_s,
            records[distance].channels[direction],
            results[key],
            alpha=ALPHA,
        )
        display(figures[key])
```

- [ ] **Step 7: Regenerate and verify GREEN**

```powershell
python tools/build_blast_individual_vmd_custom_centers_notebook.py
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "plot_vmd_modes or analyze_all_records" -v
```

Expected: both tests PASS.

- [ ] **Step 8: Commit plotting and orchestration**

```powershell
git add tools/build_blast_individual_vmd_custom_centers_notebook.py tests/test_blast_individual_vmd_custom_centers_notebook.py blast_individual_vmd_custom_centers.ipynb
git commit -m "feat: plot all custom-center VMD modes"
```

### Task 5: Run integration verification and inspect generated figures

**Files:**
- Modify: `tests/test_blast_individual_vmd_custom_centers_notebook.py`
- Modify: `tools/build_blast_individual_vmd_custom_centers_notebook.py`
- Regenerate: `blast_individual_vmd_custom_centers.ipynb`

- [ ] **Step 1: Add a Notebook regeneration stability test**

```python
def test_regeneration_is_stable():
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    first = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    assert NOTEBOOK.read_bytes() == first
```

- [ ] **Step 2: Add an executable quick-integration test**

```python
def test_notebook_executes_quick_path(tmp_path):
    env = dict(os.environ)
    env["BLAST_VMD_QUICK_TEST"] = "1"
    command = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        NOTEBOOK.name,
        "--output", str(tmp_path / "executed.ipynb"),
        "--ExecutePreprocessor.timeout=600",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
```

The quick-test branch must shorten every loaded channel to a documented small prefix (for example 512 samples) in `load-records`, while retaining all nine keys, and reduce iterations only for automated execution.

- [ ] **Step 3: Run the new tests and verify RED if quick behavior is missing**

```powershell
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -k "regeneration_is_stable or notebook_executes_quick_path" -v
```

Expected: stability PASS; quick integration initially FAIL or time out until the load-records quick truncation is implemented.

- [ ] **Step 4: Implement quick-only input truncation without changing normal runs**

```python
records = {
    distance: load_instantel_record(path)
    for distance, path in INPUT_FILES.items()
}
if QUICK_TEST:
    records = {
        key: BlastRecord(
            path=value.path,
            fs=value.fs,
            pretrigger_seconds=value.pretrigger_seconds,
            unit=value.unit,
            channels={name: channel[:512] for name, channel in value.channels.items()},
            time_s=value.time_s[:512],
        )
        for key, value in records.items()
    }
```

- [ ] **Step 5: Run the complete automated test suite**

```powershell
python tools/build_blast_individual_vmd_custom_centers_notebook.py
pytest tests/test_blast_individual_vmd_custom_centers_notebook.py -v
pytest -q
```

Expected: all new tests pass; the existing repository suite reports zero failures.

- [ ] **Step 6: Execute a full-data smoke run with practical convergence observation**

Open the Notebook, replace the nine example center lists with the user's intended values, then execute all cells. Confirm for each of the nine analyses:

- one original-plus-K-modes figure is produced;
- final centers are finite and below Nyquist;
- final centers are not mechanically identical to deliberately offset initial centers;
- modes have the same sample length as their source record;
- reconstruction RMSE is printed and finite;
- memory remains bounded because only two solver states are allocated.

If a 10000-iteration full run is unnecessarily long, retain 10000 as an upper bound but rely on tolerance-based early stopping; do not silently lower the user's configured maximum.

- [ ] **Step 7: Visually inspect all nine figures**

Check that labels are legible, no axes are clipped, every Mode line spans the full record, and the title reports the correct distance/direction/K/alpha. If a figure is too tall for large K, adjust only the per-row height; do not overlay modes.

- [ ] **Step 8: Check the final diff and commit verification updates**

```powershell
git diff --check
git status --short
git add docs/superpowers/specs/2026-07-13-blast-individual-vmd-custom-centers-design.md docs/superpowers/plans/2026-07-13-blast-individual-vmd-custom-centers.md tools/build_blast_individual_vmd_custom_centers_notebook.py tests/test_blast_individual_vmd_custom_centers_notebook.py blast_individual_vmd_custom_centers.ipynb
git commit -m "test: verify nine-signal custom-center VMD analysis"
```

Do not stage unrelated pre-existing Notebook or image modifications shown by `git status`.

## Final verification checklist

- [ ] `ALPHA` is exactly 2000.0 in both configuration and validation.
- [ ] All nine signals have independent K and complete `centers_hz` lists.
- [ ] No 0 Hz residual is inserted automatically.
- [ ] Hz-to-internal conversion divides by `fs/2`, matching the source VMD frequency axis.
- [ ] Every mode center updates after initialization.
- [ ] The solver returns the last completed parity buffer and computes convergence across the full mode array.
- [ ] All three input files retain their independent full lengths in normal execution.
- [ ] Nine figures each contain one original waveform plus K separate mode panels.
- [ ] New targeted tests and the full repository suite pass with fresh output.
- [ ] Existing unrelated dirty-worktree changes remain unstaged and unmodified.
