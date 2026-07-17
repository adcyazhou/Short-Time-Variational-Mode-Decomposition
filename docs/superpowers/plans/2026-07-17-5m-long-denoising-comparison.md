# 5m/Long Four-Signal Denoising Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate verified time-domain and direct-FFT comparisons of the raw 5m/Long blast signal, CEEMDAN, VMD-SSA, and VMD-MPE with MPE threshold 0.60.

**Architecture:** A focused Python module loads and validates all four sources, aligns them by sample index onto the original pre-trigger time axis, calculates a directly scaled one-sided FFT without windowing or filtering, and exports publication-ready figures and audit tables. Pytest tests cover parsing, alignment rejection, FFT scaling, and expected output creation before the real data pipeline is run.

**Tech Stack:** Python 3.14, NumPy, pandas, Matplotlib, pytest.

---

## File structure

- Create `tools/compare_5m_long_denoising_methods.py`: loaders, validation, direct FFT, statistics, plotting, and CLI entry point.
- Create `tests/test_compare_5m_long_denoising_methods.py`: unit and integration tests using temporary files and deterministic signals.
- Generate `output/5m_long_four_method_comparison_mpe_0_60/`: figures and CSV audit artifacts.

### Task 1: Validate and align the four signal sources

**Files:**
- Create: `tests/test_compare_5m_long_denoising_methods.py`
- Create: `tools/compare_5m_long_denoising_methods.py`

- [ ] **Step 1: Write failing loader and alignment tests**

```python
import numpy as np
import pandas as pd
import pytest

from tools.compare_5m_long_denoising_methods import align_signals


def test_align_signals_uses_original_time_and_expected_columns():
    time = np.array([-0.5, -0.25, 0.0])
    original = np.array([1.0, 2.0, 3.0])
    result = align_signals(
        time,
        original,
        np.array([1.1, 2.1, 3.1]),
        np.array([0.9, 1.9, 2.9]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert result.columns.tolist() == [
        "time_s", "original", "ceemdan", "vmd_ssa", "vmd_mpe_0_60"
    ]
    np.testing.assert_allclose(result["time_s"], time)


def test_align_signals_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number of samples"):
        align_signals(
            np.arange(3.0), np.arange(3.0), np.arange(2.0),
            np.arange(3.0), np.arange(3.0)
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py -q`

Expected: collection fails because `tools.compare_5m_long_denoising_methods` does not exist.

- [ ] **Step 3: Implement the Instantel loader, CSV loaders, and alignment validation**

Create functions with these exact interfaces:

```python
def load_instantel_long(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return original time, Long velocity, sample rate, and unit."""


def load_denoised_csv(
    path: Path, original_column: str, denoised_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CSV time, embedded original signal, and denoised signal."""


def align_signals(
    time_s: np.ndarray,
    original: np.ndarray,
    ceemdan: np.ndarray,
    vmd_ssa: np.ndarray,
    vmd_mpe: np.ndarray,
) -> pd.DataFrame:
    arrays = [time_s, original, ceemdan, vmd_ssa, vmd_mpe]
    if len({len(np.asarray(value)) for value in arrays}) != 1:
        raise ValueError("all signals must contain the same number of samples")
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays):
        raise ValueError("time and signals must contain only finite values")
    return pd.DataFrame({
        "time_s": time_s,
        "original": original,
        "ceemdan": ceemdan,
        "vmd_ssa": vmd_ssa,
        "vmd_mpe_0_60": vmd_mpe,
    })
```

`load_instantel_long` must parse the `Sample Rate`, `Pre-trigger Length`, and `Units` metadata, locate the `Tran Vert Long` header, read exactly three finite numeric columns, and construct `np.arange(n) / fs - pretrigger`. The CLI must try the user-provided cloud path first and fall back to the workspace `5m.TXT` only when the cloud file returns zero readable bytes. Each denoising CSV's embedded original column must match the parsed Long signal with `np.testing.assert_allclose(..., rtol=0, atol=0)`.

- [ ] **Step 4: Run the loader tests and verify GREEN**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py -q`

Expected: the two tests pass.

- [ ] **Step 5: Commit loader work**

```powershell
git add -- tools/compare_5m_long_denoising_methods.py tests/test_compare_5m_long_denoising_methods.py
git commit -m "feat: load and align 5m long denoising signals"
```

### Task 2: Calculate the unwindowed one-sided FFT

**Files:**
- Modify: `tests/test_compare_5m_long_denoising_methods.py`
- Modify: `tools/compare_5m_long_denoising_methods.py`

- [ ] **Step 1: Add a failing amplitude-scaling test**

```python
from tools.compare_5m_long_denoising_methods import one_sided_amplitude


def test_one_sided_amplitude_without_window_recovers_sine_amplitude():
    fs = 1024.0
    n = 1024
    time = np.arange(n) / fs
    signal = 2.5 * np.sin(2 * np.pi * 64.0 * time)
    frequency, amplitude = one_sided_amplitude(signal, fs)
    peak = np.argmax(amplitude[1:]) + 1
    assert frequency[peak] == pytest.approx(64.0)
    assert amplitude[peak] == pytest.approx(2.5, rel=1e-12)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py::test_one_sided_amplitude_without_window_recovers_sine_amplitude -q`

Expected: import fails because `one_sided_amplitude` is missing.

- [ ] **Step 3: Implement direct single-sided FFT scaling**

```python
def one_sided_amplitude(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be positive")
    spectrum = np.fft.rfft(values)
    amplitude = np.abs(spectrum) / values.size
    if values.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    frequency = np.fft.rfftfreq(values.size, d=1.0 / fs)
    return frequency, amplitude
```

This function must not subtract the mean, apply a window, pad, filter, smooth, or normalize each signal independently.

- [ ] **Step 4: Run all tests and verify GREEN**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py -q`

Expected: three tests pass.

- [ ] **Step 5: Commit FFT work**

```powershell
git add -- tools/compare_5m_long_denoising_methods.py tests/test_compare_5m_long_denoising_methods.py
git commit -m "feat: calculate direct one-sided fft amplitudes"
```

### Task 3: Export publication-ready comparison figures and audit tables

**Files:**
- Modify: `tests/test_compare_5m_long_denoising_methods.py`
- Modify: `tools/compare_5m_long_denoising_methods.py`

- [ ] **Step 1: Add a failing temporary-output integration test**

```python
from tools.compare_5m_long_denoising_methods import export_results


def test_export_results_creates_all_artifacts(tmp_path):
    fs = 1000.0
    time = np.arange(1000) / fs
    frame = pd.DataFrame({
        "time_s": time,
        "original": np.sin(2 * np.pi * 20 * time),
        "ceemdan": np.sin(2 * np.pi * 20 * time),
        "vmd_ssa": np.sin(2 * np.pi * 20 * time),
        "vmd_mpe_0_60": np.sin(2 * np.pi * 20 * time),
    })
    export_results(frame, fs, "mm/s", tmp_path)
    expected = {
        "5m_Long_time_comparison.png", "5m_Long_time_comparison.pdf",
        "5m_Long_fft_0_250Hz.png", "5m_Long_fft_0_250Hz.pdf",
        "5m_Long_time_fft_comparison.png", "5m_Long_time_fft_comparison.pdf",
        "5m_Long_four_signals_aligned.csv", "5m_Long_fft_0_250Hz.csv",
        "5m_Long_four_signal_statistics.csv",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert all(path.stat().st_size > 0 for path in tmp_path.iterdir())
```

- [ ] **Step 2: Run the integration test and verify RED**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py::test_export_results_creates_all_artifacts -q`

Expected: import fails because `export_results` is missing.

- [ ] **Step 3: Implement statistics, plotting, and exports**

Use the fixed signal order and style mapping:

```python
SERIES = {
    "original": ("Original", "#000000", "-", 0.9),
    "ceemdan": ("CEEMDAN", "#E69F00", "--", 1.0),
    "vmd_ssa": ("VMD-SSA", "#0072B2", "-.", 1.0),
    "vmd_mpe_0_60": ("VMD-MPE (threshold 0.60)", "#009E73", ":", 1.1),
}
```

Implement `export_results(frame, fs, unit, output_dir)` to:

1. Save the aligned frame as UTF-8-SIG CSV.
2. Calculate all four direct FFTs with `one_sided_amplitude`, select `frequency <= 250.0`, and save one frequency column plus four amplitude columns.
3. Save peak absolute value, mean, RMS, time-domain standard deviation, dominant frequency within 0–250 Hz, and dominant spectral amplitude for each signal.
4. Create a 12-by-5-inch time plot, a 12-by-5-inch frequency plot, and a 12-by-9-inch two-panel combination figure.
5. Label axes `Time (s)`, `Velocity (mm/s)`, `Frequency (Hz)`, and `One-sided amplitude (mm/s)`; set frequency x-limits to exactly `(0, 250)`.
6. Use the fixed colors and redundant line styles, remove top/right spines, use light grids, place unobtrusive legends, and save each figure as 300 dpi PNG plus vector PDF.

- [ ] **Step 4: Run all comparison tests and verify GREEN**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py -q`

Expected: four tests pass with no warnings or failures.

- [ ] **Step 5: Commit export work**

```powershell
git add -- tools/compare_5m_long_denoising_methods.py tests/test_compare_5m_long_denoising_methods.py
git commit -m "feat: export four-method denoising comparison"
```

### Task 4: Run on the real 5m/Long data and independently verify outputs

**Files:**
- Generate: `output/5m_long_four_method_comparison_mpe_0_60/*`

- [ ] **Step 1: Run the real-data pipeline**

Run:

```powershell
python tools/compare_5m_long_denoising_methods.py
```

Expected: exit code 0, 33,493 aligned samples reported, sample rate 4096 Hz, and nine non-empty artifacts written to `output/5m_long_four_method_comparison_mpe_0_60/`.

- [ ] **Step 2: Run the complete focused test file**

Run: `python -m pytest tests/test_compare_5m_long_denoising_methods.py -q`

Expected: all tests pass.

- [ ] **Step 3: Independently verify numeric output**

Run a separate Python assertion script that checks:

```python
assert len(aligned) == 33493
assert aligned.isna().sum().sum() == 0
assert frequency.iloc[0] == 0.0
assert frequency.iloc[-1] <= 250.0
assert np.diff(frequency).min() > 0
assert np.isclose(np.diff(frequency).mean(), 4096.0 / 33493)
assert set(statistics["signal"]) == {
    "Original", "CEEMDAN", "VMD-SSA", "VMD-MPE (threshold 0.60)"
}
```

Expected: all assertions pass and the script prints `NUMERIC_VERIFICATION=PASS`.

- [ ] **Step 4: Inspect all three PNG figures visually**

Confirm that the four legend entries are visible, time and frequency units are correct, 0–250 Hz is shown, line colors and styles remain distinguishable, and no title, legend, or axis label is clipped.

- [ ] **Step 5: Report artifacts without committing generated outputs**

Provide clickable links to the three PNG figures, their PDF versions, and the three CSV audit files. Report the precise alignment checks and the frequency resolution. Preserve all unrelated dirty-worktree files untouched.
