# Blast Multichannel STVMD Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Jupyter Notebook that loads the 5m/10m/15m blast velocity records, performs three batched dynamic multichannel STVMD decompositions, and produces the approved paper-style figures and saved results.

**Architecture:** A deterministic builder creates the final `.ipynb` so the large JSON artifact remains reproducible and reviewable. All runtime logic is embedded in tagged Notebook cells; tests load those cells directly, so the delivered Notebook has no runtime dependency on the builder or another source module. Dynamic STVMD processes independent time-window columns in bounded batches and reconstructs modes with global weighted overlap-add.

**Tech Stack:** Python 3, Jupyter/nbformat/nbclient, NumPy, SciPy, Matplotlib, pytest.

---

## File Structure

- Create `blast_multichannel_stvmd.ipynb`: self-contained user-facing analysis with Markdown explanations, configuration, algorithm, three decompositions, figures, and exports.
- Create `tools/build_blast_multichannel_stvmd_notebook.py`: deterministic source for Notebook cells; running it rewrites only the target Notebook.
- Create `tests/test_blast_multichannel_stvmd_notebook.py`: unit, numerical, plotting, and Notebook smoke tests that execute tagged Notebook cells.
- Modify `.gitignore`: ignore `.superpowers/`, `tmp/`, generated Notebook checkpoints, and `output/stvmd_blast/`; do not ignore the three input TXT files.

### Task 1: Create the Notebook scaffold and verified Instantel data loader

**Files:**
- Create: `tools/build_blast_multichannel_stvmd_notebook.py`
- Create: `blast_multichannel_stvmd.ipynb`
- Create: `tests/test_blast_multichannel_stvmd_notebook.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing loader and structure tests**

Create `tests/test_blast_multichannel_stvmd_notebook.py` with a helper that executes only cells tagged `core`:

```python
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
        "参数配置", "读取与校验", "动态多通道 STVMD",
        "Tran 方向", "Vert 方向", "Long 方向", "结果保存",
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
        record.data,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
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
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: FAIL because `blast_multichannel_stvmd.ipynb` does not exist.

- [ ] **Step 3: Implement the deterministic Notebook builder and loader cells**

Create `tools/build_blast_multichannel_stvmd_notebook.py` using `nbformat.v4`. The builder must add the approved title and section Markdown cells, a tagged `core` imports cell, and a tagged `core` loader cell containing:

```python
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import warnings

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BlastRecord:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    columns: tuple
    data: np.ndarray


def _metadata_number(metadata, key):
    if key not in metadata:
        raise ValueError(f"缺少元数据字段: {key}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", metadata[key])
    if match is None:
        raise ValueError(f"无法解析元数据字段 {key}: {metadata[key]!r}")
    return float(match.group())


def load_instantel_txt(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    metadata = {}
    header_index = None
    for index, raw in enumerate(lines):
        stripped = raw.strip().strip('"')
        if all(name in stripped for name in ("Tran", "Vert", "Long")):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()
    if header_index is None:
        raise ValueError(f"{path.name}: 未找到 Tran/Vert/Long 数据表头")
    data = np.loadtxt(lines[header_index + 1 :], dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] != 3:
        raise ValueError(f"{path.name}: 期望3列数据，实际为{data.shape[1]}列")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: 数据包含 NaN 或无穷值")
    fs = _metadata_number(metadata, "Sample Rate")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    if fs <= 0:
        raise ValueError(f"{path.name}: 采样率必须为正数")
    return BlastRecord(
        path=path,
        metadata=metadata,
        fs=fs,
        pretrigger_seconds=pretrigger,
        columns=("Tran", "Vert", "Long"),
        data=data,
    )


def prepare_direction_inputs(records):
    order = ("5m", "10m", "15m")
    missing = [key for key in order if key not in records]
    if missing:
        raise ValueError(f"缺少测点记录: {missing}")
    fs_values = np.array([records[key].fs for key in order], dtype=float)
    pre_values = np.array(
        [records[key].pretrigger_seconds for key in order], dtype=float
    )
    if not np.allclose(fs_values, fs_values[0]):
        raise ValueError(f"采样率不一致: {fs_values.tolist()}")
    if not np.allclose(pre_values, pre_values[0]):
        raise ValueError(f"预触发长度不一致: {pre_values.tolist()}")
    common_n = min(records[key].data.shape[0] for key in order)
    signals = {}
    for column_index, direction in enumerate(("Tran", "Vert", "Long")):
        signals[direction] = np.vstack(
            [records[key].data[:common_n, column_index] for key in order]
        )
    time_s = np.arange(common_n) / fs_values[0] - pre_values[0]
    return signals, time_s
```

The builder writes the Notebook with:

```python
target = Path(__file__).resolve().parents[1] / "blast_multichannel_stvmd.ipynb"
nbformat.write(notebook, target)
print(f"Wrote {target}")
```

Run the builder:

```powershell
python tools/build_blast_multichannel_stvmd_notebook.py
```

- [ ] **Step 4: Add generated-artifact ignores**

Append these exact entries to `.gitignore`:

```gitignore

# Codex brainstorming and generated STVMD output
.superpowers/
tmp/
output/stvmd_blast/
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit the scaffold**

```powershell
git add .gitignore tools/build_blast_multichannel_stvmd_notebook.py blast_multichannel_stvmd.ipynb tests/test_blast_multichannel_stvmd_notebook.py
git commit -m "feat: scaffold blast STVMD notebook"
```

### Task 2: Implement memory-bounded dynamic multichannel STVMD

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] **Step 1: Write failing numerical and validation tests**

Append:

```python
import pytest


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
        x, fs=128, K=3, alpha=50.0, window_length=32,
        tau=1e-5, tol=1e-6, max_iters=80, batch_windows=17,
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
        fs=128, K=3, alpha=50.0, window_length=32,
        tau=1e-5, tol=1e-7, max_iters=100,
    )
    whole = ns["run_dynamic_stvmd_batched"](x, batch_windows=96, **kwargs)
    split = ns["run_dynamic_stvmd_batched"](x, batch_windows=13, **kwargs)
    np.testing.assert_allclose(
        split["center_freq_hz"], whole["center_freq_hz"], atol=2e-3
    )
    np.testing.assert_allclose(split["modes"], whole["modes"], atol=2e-3)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: FAIL with missing `validate_config` and `run_dynamic_stvmd_batched`.

- [ ] **Step 3: Add configuration validation and window extraction**

Add a tagged `core` cell with:

```python
REPOSITORY_WINDOWS = (8, 16, 32, 64, 128, 256)


def validate_config(K, alpha, window_length, batch_windows, max_iters):
    if K not in (3, 4, 5):
        raise ValueError("K 必须为 3、4 或 5")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ALPHA 必须为有限正数")
    if window_length not in REPOSITORY_WINDOWS:
        raise ValueError(
            f"WINDOW_LENGTH 必须来自 {REPOSITORY_WINDOWS}"
        )
    if batch_windows < 1:
        raise ValueError("BATCH_WINDOWS 必须大于0")
    if max_iters < 2:
        raise ValueError("MAX_ITERS 必须至少为2")


def _pad_width(window_length):
    left = window_length // 2
    right = window_length - 1 - left
    return left, right


def _window_batch(x_padded, window_length, start, stop, window):
    views = np.lib.stride_tricks.sliding_window_view(
        x_padded, window_shape=window_length, axis=1
    )
    segments = np.moveaxis(views[:, start:stop, :], 1, 2)
    return segments * window[None, :, None]
```

- [ ] **Step 4: Add the dynamic ADMM batch solver**

Add `_solve_dynamic_batch` using two iteration buffers:

```python
def _solve_dynamic_batch(
    f_hat, K, alpha, tau, tol, max_iters
):
    channels, freq_bins, batch_n = f_hat.shape
    normalized_freq = np.arange(1, freq_bins + 1, dtype=float) / freq_bins
    u = np.zeros(
        (2, channels, freq_bins, K, batch_n), dtype=np.complex128
    )
    lagrange = np.zeros(
        (2, channels, freq_bins, batch_n), dtype=np.complex128
    )
    omega = np.zeros((2, K, batch_n), dtype=float)
    for mode in range(K):
        omega[0, mode, :] = mode / K

    converged = False
    final_diff = np.inf
    for iteration in range(max_iters):
        current = iteration % 2
        updated = (iteration + 1) % 2
        u[updated].fill(0)
        omega[updated] = omega[current]
        running_sum = np.sum(u[current], axis=2)
        for mode in range(K):
            running_sum -= u[current, :, :, mode, :]
            denominator = 1.0 + alpha * (
                normalized_freq[:, None] - omega[current, mode, :][None, :]
            ) ** 2
            u[updated, :, :, mode, :] = (
                f_hat - running_sum - lagrange[current] / 2.0
            ) / denominator[None, :, :]
            running_sum += u[updated, :, :, mode, :]
            if mode == 0:
                omega[updated, mode, :] = 0.0
            else:
                mode_power = np.sum(
                    np.abs(u[updated, :, :, mode, :]) ** 2, axis=0
                )
                denominator_power = np.sum(mode_power, axis=0)
                numerator_power = np.sum(
                    normalized_freq[:, None] * mode_power, axis=0
                )
                omega[updated, mode, :] = np.divide(
                    numerator_power,
                    denominator_power,
                    out=np.zeros_like(numerator_power),
                    where=denominator_power > np.finfo(float).eps,
                )
        lagrange[updated] = lagrange[current] + tau * (
            np.sum(u[updated], axis=2) - f_hat
        )
        final_diff = float(np.max(np.mean(
            np.abs(u[updated] - u[current]) ** 2,
            axis=(1, 2),
        )))
        if iteration >= 2 and final_diff < tol:
            converged = True
            break

    final_index = (iteration + 1) % 2
    u_final = u[final_index]
    omega_final = omega[final_index]
    for column in range(batch_n):
        order = np.argsort(omega_final[:, column])
        sorted_modes = u_final[:, :, :, column][:, :, order].copy()
        u_final[:, :, :, column] = sorted_modes
        omega_final[:, column] = omega_final[order, column]
    return u_final, omega_final, iteration + 1, converged, final_diff
```

- [ ] **Step 5: Add batched orchestration and weighted overlap-add**

Implement:

```python
def run_dynamic_stvmd_batched(
    x, fs, K=4, alpha=50.0, window_length=64,
    tau=1e-5, tol=1e-9, max_iters=2000, batch_windows=256,
):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("输入必须为 (通道, 时间) 二维数组")
    if not np.isfinite(x).all():
        raise ValueError("输入包含 NaN 或无穷值")
    validate_config(K, alpha, window_length, batch_windows, max_iters)
    channels, sample_n = x.shape
    if sample_n < window_length:
        raise ValueError("样本数不能小于 WINDOW_LENGTH")

    left, right = _pad_width(window_length)
    x_padded = np.pad(x, ((0, 0), (left, right)), mode="reflect")
    window = scipy.signal.windows.hamming(window_length, sym=False)
    freq_bins = window_length // 2 + 1
    modes_accum = np.zeros(
        (K, channels, sample_n + window_length - 1), dtype=float
    )
    norm = np.zeros(sample_n + window_length - 1, dtype=float)
    center_freq_hz = np.zeros((K, sample_n), dtype=float)
    mean_tf_power = np.zeros((freq_bins, sample_n), dtype=float)
    iterations = []
    convergence = []

    for start in range(0, sample_n, batch_windows):
        stop = min(sample_n, start + batch_windows)
        windowed = _window_batch(
            x_padded, window_length, start, stop, window
        )
        f_hat = scipy.fft.rfft(windowed, axis=1, workers=-1)
        mean_tf_power[:, start:stop] = np.mean(np.abs(f_hat) ** 2, axis=0)
        u_hat, omega, count, converged, diff = _solve_dynamic_batch(
            f_hat, K, alpha, tau, tol, max_iters
        )
        center_freq_hz[:, start:stop] = omega * (fs / 2.0)
        batch_modes = scipy.fft.irfft(
            u_hat, n=window_length, axis=1, workers=-1
        ).real
        for local_column, global_column in enumerate(range(start, stop)):
            target = slice(global_column, global_column + window_length)
            norm[target] += window ** 2
            for mode in range(K):
                modes_accum[mode, :, target] += (
                    batch_modes[:, :, mode, local_column] * window[None, :]
                )
        iterations.append(count)
        convergence.append((converged, diff))

    safe_norm = np.where(norm > np.finfo(float).eps, norm, 1.0)
    modes_full = modes_accum / safe_norm[None, None, :]
    modes = modes_full[:, :, left : left + sample_n]
    return {
        "modes": modes,
        "center_freq_hz": center_freq_hz,
        "mean_tf_power": mean_tf_power,
        "iterations": np.asarray(iterations, dtype=int),
        "converged": np.asarray([item[0] for item in convergence], dtype=bool),
        "final_diff": np.asarray([item[1] for item in convergence], dtype=float),
    }
```

Regenerate:

```powershell
python tools/build_blast_multichannel_stvmd_notebook.py
```

- [ ] **Step 6: Run numerical tests and correct only implementation defects**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass. If a numerical tolerance fails, inspect reconstruction and convergence first; do not loosen tolerances until the update or overlap-add logic is verified.

- [ ] **Step 7: Commit the STVMD kernel**

```powershell
git add tools/build_blast_multichannel_stvmd_notebook.py blast_multichannel_stvmd.ipynb tests/test_blast_multichannel_stvmd_notebook.py
git commit -m "feat: add batched dynamic multichannel STVMD"
```

### Task 3: Add diagnostics and frequency-band mapping

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] **Step 1: Write failing diagnostic tests**

Append:

```python
def test_summarize_result_reports_reconstruction_and_bands():
    ns = notebook_namespace()
    x = synthetic_multichannel(n=128)
    raw = ns["run_dynamic_stvmd_batched"](
        x, fs=128, K=3, alpha=50.0, window_length=32,
        tau=1e-5, tol=1e-6, max_iters=80, batch_windows=19,
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
```

- [ ] **Step 2: Run the tests and verify the missing-function failures**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: FAIL with missing `summarize_stvmd_result` and `power_to_db`.

- [ ] **Step 3: Implement stable dB conversion and spectral quantiles**

Add:

```python
def power_to_db(power, floor_db=-100.0):
    power = np.asarray(power, dtype=float)
    peak = float(np.max(power))
    if not np.isfinite(peak) or peak <= 0:
        return np.full_like(power, floor_db)
    db = 10.0 * np.log10(np.maximum(power / peak, 10 ** (floor_db / 10)))
    return np.maximum(db, floor_db)


def _energy_band(freq_hz, power, low=0.05, high=0.95):
    power = np.maximum(np.asarray(power, dtype=float), 0.0)
    total = float(np.sum(power))
    if total <= np.finfo(float).eps:
        return np.array([0.0, 0.0])
    cumulative = np.cumsum(power) / total
    return np.array([
        np.interp(low, cumulative, freq_hz),
        np.interp(high, cumulative, freq_hz),
    ])
```

- [ ] **Step 4: Implement reconstruction, NRMSE, energy fractions, and bands**

Add:

```python
def summarize_stvmd_result(x, fs, raw_result):
    modes = raw_result["modes"]
    reconstruction = np.sum(modes, axis=0)
    denominator = np.linalg.norm(x, axis=1)
    nrmse = np.divide(
        np.linalg.norm(x - reconstruction, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    mode_energy = np.sum(modes ** 2, axis=2)
    channel_energy = np.sum(mode_energy, axis=0, keepdims=True)
    energy_fraction = np.divide(
        mode_energy,
        channel_energy,
        out=np.zeros_like(mode_energy),
        where=channel_energy > np.finfo(float).eps,
    )
    freq_hz = scipy.fft.rfftfreq(modes.shape[-1], d=1.0 / fs)
    bands = np.zeros((modes.shape[0], 2), dtype=float)
    mode_power = np.zeros((modes.shape[0], freq_hz.size), dtype=float)
    for mode in range(modes.shape[0]):
        spectra = scipy.fft.rfft(modes[mode], axis=1, workers=-1)
        mode_power[mode] = np.sum(np.abs(spectra) ** 2, axis=0)
        bands[mode] = _energy_band(freq_hz, mode_power[mode])
    result = dict(raw_result)
    result.update({
        "reconstruction": reconstruction,
        "nrmse": nrmse,
        "energy_fraction": energy_fraction,
        "frequency_hz": freq_hz,
        "mode_power": mode_power,
        "frequency_bands_hz": bands,
    })
    return result
```

Regenerate and run:

```powershell
python tools/build_blast_multichannel_stvmd_notebook.py
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit diagnostics**

```powershell
git add tools/build_blast_multichannel_stvmd_notebook.py blast_multichannel_stvmd.ipynb tests/test_blast_multichannel_stvmd_notebook.py
git commit -m "feat: add STVMD diagnostics and frequency bands"
```

### Task 4: Implement the approved paper-style figures

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] **Step 1: Write failing plotting tests**

Append:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def diagnostic_fixture(ns):
    x = synthetic_multichannel(n=128)
    raw = ns["run_dynamic_stvmd_batched"](
        x, fs=128, K=3, alpha=50.0, window_length=32,
        tau=1e-5, tol=1e-6, max_iters=80, batch_windows=19,
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
```

- [ ] **Step 2: Run the plotting test and verify missing functions**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: FAIL with missing plotting functions.

- [ ] **Step 3: Add shared plotting conventions**

Embed:

```python
DISTANCE_LABELS = ("5 m", "10 m", "15 m")
MODE_COLORS = ("#64748b", "#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _frequency_extent(time_s, fs, frequency_bins):
    return [time_s[0], time_s[-1], 0.0, fs / 2.0]


def _limit_frequency_axis(axis, plot_max_hz, fs):
    axis.set_ylim(0.0, min(float(plot_max_hz), fs / 2.0))
    axis.set_ylabel("Frequency (Hz)")
```

Use a colorblind-safe palette, `constrained_layout=True`, trigger lines at `t=0`, velocity labels `Velocity (mm/s)`, and consistent colors for each mode across all figures.

- [ ] **Step 4: Implement all four plotting functions**

Implement exact signatures from the test. Required content:

```python
def plot_input_and_tf(direction, x, time_s, fs, result, plot_max_hz):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for channel, label in enumerate(DISTANCE_LABELS):
        axes[0].plot(time_s, x[channel], lw=0.8, label=label)
    axes[0].axvline(0, color="black", ls="--", lw=0.8)
    axes[0].set(xlabel="Time (s)", ylabel="Velocity (mm/s)",
                title=f"{direction}: input velocity")
    axes[0].legend(frameon=False)
    image = axes[1].imshow(
        power_to_db(result["mean_tf_power"]),
        origin="lower", aspect="auto",
        extent=_frequency_extent(time_s, fs, result["mean_tf_power"].shape[0]),
        cmap="viridis", vmin=-80, vmax=0,
    )
    _limit_frequency_axis(axes[1], plot_max_hz, fs)
    axes[1].set(xlabel="Time (s)", title="Mean multichannel TF power")
    fig.colorbar(image, ax=axes[1], label="Relative power (dB)")
    return fig


def plot_modes(direction, time_s, result):
    modes = result["modes"]
    mode_n, channel_n, _ = modes.shape
    fig, axes = plt.subplots(
        mode_n, channel_n,
        figsize=(12, max(4.0, 2.1 * mode_n)),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for mode in range(mode_n):
        row_name = "Residual" if mode == 0 else f"Mode {mode}"
        for channel in range(channel_n):
            axis = axes[mode, channel]
            axis.plot(
                time_s, modes[mode, channel],
                color=MODE_COLORS[mode], lw=0.7,
            )
            axis.axvline(0, color="black", ls="--", lw=0.6)
            if mode == 0:
                axis.set_title(DISTANCE_LABELS[channel])
            if channel == 0:
                axis.set_ylabel(f"{row_name}\nVelocity (mm/s)")
            if mode == mode_n - 1:
                axis.set_xlabel("Time (s)")
    fig.suptitle(f"{direction}: dynamic STVMD modes")
    return fig


def plot_if_and_reconstruction(
    direction, x, time_s, result, plot_max_hz
):
    mode_n, channel_n, _ = result["modes"].shape
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.3, 0.8))
    axis_if = fig.add_subplot(grid[0, :])
    for mode in range(1, mode_n):
        axis_if.plot(
            time_s, result["center_freq_hz"][mode],
            color=MODE_COLORS[mode], lw=1.1, label=f"Mode {mode}",
        )
    axis_if.axvline(0, color="black", ls="--", lw=0.7)
    axis_if.set(
        xlabel="Time (s)", ylabel="Frequency (Hz)",
        title=f"{direction}: instantaneous center frequencies",
        ylim=(0.0, plot_max_hz),
    )
    axis_if.legend(frameon=False, ncol=max(1, mode_n - 1))

    for channel in range(channel_n):
        axis = fig.add_subplot(grid[1, channel])
        axis.plot(time_s, x[channel], color="#64748b", lw=0.8, label="Input")
        axis.plot(
            time_s, result["reconstruction"][channel],
            color="#D55E00", lw=0.7, alpha=0.85, label="Reconstruction",
        )
        axis.axvline(0, color="black", ls="--", lw=0.6)
        axis.set(
            xlabel="Time (s)", ylabel="Velocity (mm/s)",
            title=f"{DISTANCE_LABELS[channel]}  NRMSE={result['nrmse'][channel]:.3g}",
        )
        if channel == 0:
            axis.legend(frameon=False)

    axis_energy = fig.add_subplot(grid[2, :])
    energy_image = axis_energy.imshow(
        result["energy_fraction"], aspect="auto", cmap="magma",
        vmin=0.0, vmax=max(1e-12, float(result["energy_fraction"].max())),
    )
    axis_energy.set(
        xlabel="Channel", ylabel="Component", title="Mode energy fraction"
    )
    axis_energy.set_xticks(range(channel_n), DISTANCE_LABELS[:channel_n])
    axis_energy.set_yticks(
        range(mode_n),
        ["Residual"] + [f"Mode {mode}" for mode in range(1, mode_n)],
    )
    fig.colorbar(energy_image, ax=axis_energy, label="Fraction")
    return fig


def plot_spectrum_if_mapping(
    direction, x, time_s, fs, result, plot_max_hz
):
    freq_hz = scipy.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    spectra = scipy.fft.rfft(x, axis=1, workers=-1)
    combined_amplitude = np.sqrt(np.mean(np.abs(spectra) ** 2, axis=0))
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.2),
        gridspec_kw={"width_ratios": (1.0, 3.2)},
        sharey=True, constrained_layout=True,
    )
    axes[0].plot(combined_amplitude, freq_hz, color="#475569", lw=0.8)
    axes[0].set(
        xlabel="Combined amplitude", ylabel="Frequency (Hz)",
        title="Fourier spectrum",
    )
    tf_image = axes[1].imshow(
        power_to_db(result["mean_tf_power"]),
        origin="lower", aspect="auto",
        extent=_frequency_extent(time_s, fs, result["mean_tf_power"].shape[0]),
        cmap="viridis", vmin=-80, vmax=0,
    )
    for mode in range(1, result["modes"].shape[0]):
        color = MODE_COLORS[mode]
        band_low, band_high = result["frequency_bands_hz"][mode]
        axes[1].plot(
            time_s, result["center_freq_hz"][mode],
            color=color, lw=1.2, label=f"Mode {mode}: {band_low:.1f}-{band_high:.1f} Hz",
        )
        for boundary in (band_low, band_high):
            axes[0].axhline(boundary, color=color, ls="--", lw=0.8)
            axes[1].axhline(boundary, color=color, ls="--", lw=0.8)
    axes[1].axvline(0, color="white", ls=":", lw=0.8)
    axes[1].set(xlabel="Time (s)", title=f"{direction}: TF spectrum and IF tracks")
    axes[1].legend(frameon=True, fontsize=8, loc="upper right")
    _limit_frequency_axis(axes[0], plot_max_hz, fs)
    _limit_frequency_axis(axes[1], plot_max_hz, fs)
    fig.colorbar(tf_image, ax=axes[1], label="Relative power (dB)")
    return fig
```

The functions above create `K × 3` mode axes, show nonzero center-frequency tracks, compare each channel with its reconstruction, report NRMSE and energy fractions, and draw each mode’s 5%/95% band boundaries on both sides of the spectrum/IF mapping figure.

Regenerate and test:

```powershell
python tools/build_blast_multichannel_stvmd_notebook.py
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass with no Matplotlib warnings.

- [ ] **Step 5: Commit figures**

```powershell
git add tools/build_blast_multichannel_stvmd_notebook.py blast_multichannel_stvmd.ipynb tests/test_blast_multichannel_stvmd_notebook.py
git commit -m "feat: add paper-style blast STVMD figures"
```

### Task 5: Assemble three-direction execution, exports, and Notebook smoke test

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] **Step 1: Write failing export and smoke tests**

Append:

```python
import json
import os
import subprocess
import sys


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
        tmp_path, {"Tran": result},
        {"K": 3, "ALPHA": 50.0, "WINDOW_LENGTH": 32},
    )
    assert (tmp_path / "stvmd_results.npz").is_file()
    assert len(list(tmp_path.glob("tran_*.png"))) == 4
    for figure in figures.values():
        plt.close(figure)


def test_notebook_executes_in_quick_test_mode(tmp_path):
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
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, capture_output=True, text=True, timeout=360,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NOTEBOOK_SMOKE_OK" in completed.stdout
```

- [ ] **Step 2: Run the tests and verify expected failures**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: FAIL with missing `save_direction_figures`, `save_all_results`, and incomplete execution cells.

- [ ] **Step 3: Add quick-test-aware configuration and full data cells**

The configuration cell must default to the approved values and only reduce workload when the environment variable is explicitly set:

```python
QUICK_TEST = os.environ.get("STVMD_QUICK_TEST") == "1"

K = 3 if QUICK_TEST else 4
ALPHA = 50.0
WINDOW_LENGTH = 16 if QUICK_TEST else 64
TAU = 1e-5
TOL = 1e-6 if QUICK_TEST else 1e-9
MAX_ITERS = 20 if QUICK_TEST else 2000
BATCH_WINDOWS = 64 if QUICK_TEST else 256
PLOT_MAX_HZ = 200.0
SAVE_OUTPUTS = False if QUICK_TEST else True
```

Add visible cells that load `5m.TXT`, `10m.TXT`, `15m.TXT`, print metadata and common duration, and limit each direction to 256 samples only in quick-test mode.

- [ ] **Step 4: Add the explicit Tran, Vert, and Long execution cells**

Each direction gets its own Markdown heading and code cell:

```python
def analyze_direction(direction, x, time_s, fs):
    raw = run_dynamic_stvmd_batched(
        x, fs=fs, K=K, alpha=ALPHA, window_length=WINDOW_LENGTH,
        tau=TAU, tol=TOL, max_iters=MAX_ITERS,
        batch_windows=BATCH_WINDOWS,
    )
    result = summarize_stvmd_result(x, fs, raw)
    if not np.all(result["converged"]):
        failed = np.flatnonzero(~result["converged"])
        warnings.warn(
            f"{direction}: 批次 {failed.tolist()} 达到最大迭代数；"
            f"最大最终差值={result['final_diff'][failed].max():.3e}"
        )
    figures = {
        "input_tf": plot_input_and_tf(
            direction, x, time_s, fs, result, PLOT_MAX_HZ
        ),
        "modes": plot_modes(direction, time_s, result),
        "if_reconstruction": plot_if_and_reconstruction(
            direction, x, time_s, result, PLOT_MAX_HZ
        ),
        "spectrum_if_mapping": plot_spectrum_if_mapping(
            direction, x, time_s, fs, result, PLOT_MAX_HZ
        ),
    }
    return result, figures
```

Then call it separately for `Tran`, `Vert`, and `Long`, assigning into `results` and `figures_by_direction`.

- [ ] **Step 5: Implement deterministic output saving**

Add:

```python
def save_direction_figures(output_dir, direction, figures):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = direction.lower()
    for name, figure in figures.items():
        figure.savefig(
            output_dir / f"{prefix}_{name}.png",
            dpi=300, bbox_inches="tight",
        )


def save_all_results(output_dir, results, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"config_json": json.dumps(config, ensure_ascii=False)}
    for direction, result in results.items():
        prefix = direction.lower()
        for key in (
            "modes", "center_freq_hz", "reconstruction", "nrmse",
            "energy_fraction", "frequency_bands_hz",
        ):
            arrays[f"{prefix}_{key}"] = result[key]
    np.savez_compressed(
        output_dir / "stvmd_results.npz",
        **arrays,
    )
```

The final Notebook cell saves four PNGs per direction and one combined `stvmd_results.npz` to `output/stvmd_blast/` only when `SAVE_OUTPUTS` is true.

- [ ] **Step 6: Regenerate and run the complete test suite**

Run:

```powershell
python tools/build_blast_multichannel_stvmd_notebook.py
python -m pytest -q
```

Expected: all tests pass, including `NOTEBOOK_SMOKE_OK`.

- [ ] **Step 7: Inspect Notebook structure and generated figures**

Run:

```powershell
python -c "import nbformat; n=nbformat.read('blast_multichannel_stvmd.ipynb',as_version=4); print(n.nbformat, len(n.cells)); print([c.source.splitlines()[0] for c in n.cells if c.cell_type=='markdown'])"
```

Expected: nbformat 4; Markdown headings include configuration, loading, algorithm, all three directions, and saving.

Execute quick-test mode once and inspect its four figure families for clipped labels, incorrect units, missing legends, and mismatched frequency colors. Correct layout defects in the builder, regenerate, and rerun `python -m pytest -q`.

- [ ] **Step 8: Commit the complete Notebook**

```powershell
git add tools/build_blast_multichannel_stvmd_notebook.py blast_multichannel_stvmd.ipynb tests/test_blast_multichannel_stvmd_notebook.py
git commit -m "feat: complete blast multichannel STVMD notebook"
```

### Task 6: Final verification against the approved design

**Files:**
- Verify: `blast_multichannel_stvmd.ipynb`
- Verify: `tests/test_blast_multichannel_stvmd_notebook.py`
- Verify: `docs/superpowers/specs/2026-07-04-blast-multichannel-stvmd-design.md`

- [ ] **Step 1: Run fresh automated verification**

```powershell
python -m pytest -q
python tools/build_blast_multichannel_stvmd_notebook.py
git diff --exit-code -- blast_multichannel_stvmd.ipynb
git diff --check
```

Expected: tests pass; rebuilding does not change the Notebook; no whitespace errors.

- [ ] **Step 2: Verify the real input contract without full decomposition**

```powershell
python -c "import nbformat; n=nbformat.read('blast_multichannel_stvmd.ipynb',as_version=4); ns={'__name__':'verify'}; [exec(compile(c.source,'blast_multichannel_stvmd.ipynb','exec'),ns) for c in n.cells if c.cell_type=='code' and 'core' in c.metadata.get('tags',[])]; r={d:ns['load_instantel_txt'](f'{d}.TXT') for d in ('5m','10m','15m')}; s,t=ns['prepare_direction_inputs'](r); print(r['5m'].fs, {k:v.shape for k,v in s.items()}, t[0], t[-1])"
```

Expected:

```text
4096.0 {'Tran': (3, 14336), 'Vert': (3, 14336), 'Long': (3, 14336)} -0.5 2.999755859375
```

- [ ] **Step 3: Review requirement coverage**

Confirm directly in the Notebook:

- three explicit decompositions in Tran → Vert → Long order;
- channels ordered 5m → 10m → 15m;
- common-length truncation, no zero padding;
- Hamming window, reflection padding, hop length 1;
- K 3–5, alpha default 50, repository window choices;
- modes and non-smoothed dynamic center-frequency plots;
- mean multichannel power TF spectrum;
- spectrum/IF mapping with shared colors and 5%–95% bands;
- velocity units throughout;
- optional 300 dpi PNG and compressed NPZ export;
- detailed Chinese Markdown explanations and executable cells.

- [ ] **Step 4: Report measured verification**

Record the exact pytest pass count, quick Notebook execution status, deterministic rebuild result, and real-input shapes in the handoff. Do not claim the full 14336-point decomposition was executed unless it was actually run.
