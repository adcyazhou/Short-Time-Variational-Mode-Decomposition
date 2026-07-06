# VMD Center-Frequency Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the inherited VMD convergence index defect and replace misleading FFT display figures with scientifically verified center-frequency-versus-time figures.

**Architecture:** The notebook generator continues to source algorithm code from `main_STVMD.ipynb`, but applies one guarded, audited correction to the VMD convergence line. Shared plotting accepts scalar VMD centers or time-varying STVMD centers and produces method-appropriate curves without hiding frequencies by default.

**Tech Stack:** Python, NumPy, SciPy, Matplotlib, nbformat, nbclient, pytest.

---

### Task 1: Guard and Correct the VMD Convergence Index

**Files:**
- Modify: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Write the failing source-audit test**

Replace the current all-verbatim assertion with:

```python
VMD_BUGGY_DELTA = (
    "delta = u_hat_plus[0,:,i]-u_hat_plus[1,:,i]"
)
VMD_CORRECTED_DELTA = (
    "delta = u_hat_plus[0,:,:,i]-u_hat_plus[1,:,:,i]"
)


def test_algorithm_sources_preserve_only_the_audited_vmd_correction():
    generated = nbformat.read(NOTEBOOK, as_version=4)
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    generated_buffer = find_source_cell(
        generated, ("def buffer(", "def unbuffer(", "def window_norm(")
    )
    generated_vmd = find_source_cell(generated, ("class VMD(object):",))
    generated_stvmd = find_source_cell(
        generated, ("class STVMD(object):",)
    )
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
```

- [ ] **Step 2: Run the source-audit test and verify RED**

Run:

```powershell
python -m pytest `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py::test_algorithm_sources_preserve_only_the_audited_vmd_correction `
  -q
```

Expected: FAIL because the generated VMD still contains the buggy convergence
index.

- [ ] **Step 3: Add the guarded source correction**

After extracting the source strings in the generator, add:

```python
VMD_BUGGY_DELTA = (
    "delta = u_hat_plus[0,:,i]-u_hat_plus[1,:,i]"
)
VMD_CORRECTED_DELTA = (
    "delta = u_hat_plus[0,:,:,i]-u_hat_plus[1,:,:,i]"
)
if VMD_SOURCE.count(VMD_BUGGY_DELTA) != 1:
    raise RuntimeError(
        "Expected exactly one known VMD convergence-index defect"
    )
VMD_SOURCE = VMD_SOURCE.replace(
    VMD_BUGGY_DELTA, VMD_CORRECTED_DELTA
)
```

Do not modify `BUFFER_SOURCE` or `STVMD_SOURCE`.

- [ ] **Step 4: Mark the corrected source explicitly**

Change the generated VMD cell to:

```python
code(
    VMD_SOURCE,
    "corrected-vmd-source",
    tags=("core", "corrected-algorithm-source"),
),
```

Update the deterministic cell-contract test to expect this ID and tag. Keep
the buffer and STVMD cells tagged `original-algorithm-source`.

- [ ] **Step 5: Add the known-frequency scientific regression**

Add:

```python
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
```

- [ ] **Step 6: Regenerate and verify GREEN**

Run:

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "audited_vmd or known_20_and_28" -q
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "fix: correct VMD convergence indexing"
```

### Task 2: Replace FFT Figures with Center-Frequency Curves

**Files:**
- Modify: `tools/build_single_waveform_vmd_stvmd_original_notebook.py`
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Change the safe default and validation tests**

Change the parameter contract to require:

```python
"PLOT_MAX_HZ = None"
```

Add a validation test proving `None` is accepted and non-positive explicit
limits are rejected.

- [ ] **Step 2: Write failing center-frequency plot tests**

Add:

```python
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
    for axis, expected in zip(
        figure.axes, result["center_frequency_hz"]
    ):
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
    matplotlib.pyplot.close(figure)


def test_automatic_center_frequency_limits_do_not_hide_high_modes():
    namespace = notebook_namespace()
    time_s, fs, result = synthetic_result(namespace)
    result["center_frequency_hz"] = np.array(
        [0.0, 50.0, 215.0, 1378.0]
    )
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
```

- [ ] **Step 3: Run plot tests and verify RED**

Run:

```powershell
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "center_frequency_plot or automatic_center_frequency_limits" -q
```

Expected: FAIL because `plot_center_frequencies` does not exist.

- [ ] **Step 4: Implement center-frequency normalization and plotting**

Replace `plot_modal_frequency` with:

```python
def center_frequency_tracks(time_s, center_frequency_hz):
    centers = np.asarray(center_frequency_hz, dtype=float)
    if centers.ndim == 1:
        return np.repeat(centers[:, None], len(time_s), axis=1)
    if centers.ndim != 2 or centers.shape[1] != len(time_s):
        raise ValueError(
            "Center frequencies must have shape (K,) or (K, time)"
        )
    return centers


def plot_center_frequencies(
    method, time_s, result, plot_max_hz=None
):
    tracks = center_frequency_tracks(
        time_s, result["center_frequency_hz"]
    )
    figure, axes = plt.subplots(
        tracks.shape[0],
        1,
        figsize=(11, 2.0 * tracks.shape[0]),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(time_s, tracks[index], lw=1.0)
        label = (
            "Residual (fixed 0 Hz)"
            if index == 0
            else f"Mode {index}"
        )
        axis.set_ylabel(f"{label}\nFrequency (Hz)")
        if plot_max_hz is not None:
            axis.set_ylim(0.0, float(plot_max_hz))
    axes[0].set_title(f"{method}: center frequencies")
    axes[-1].set_xlabel("Time (s)")
    return figure
```

Change `plot_method_results` to return:

```python
{
    "time_modes": plot_modal_time(method, time_s, result),
    "center_frequencies": plot_center_frequencies(
        method, time_s, result, plot_max_hz
    ),
    "energy_fraction": plot_energy_fraction(method, result),
}
```

- [ ] **Step 5: Allow automatic plot limits in validation**

Remove `PLOT_MAX_HZ` from the unconditional positive-number loop and add:

```python
if plot_max_hz is not None and (
    not np.isfinite(plot_max_hz) or plot_max_hz <= 0
):
    raise ValueError(
        "PLOT_MAX_HZ must be None or a finite positive number"
    )
```

- [ ] **Step 6: Update figure contracts and verify GREEN**

Update figure keys and axis checks from `frequency_modes` to
`center_frequencies`. Update the pipeline marker from
`def plot_modal_frequency` to `def plot_center_frequencies`.

Run:

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest tests/test_single_waveform_vmd_stvmd_original_notebook.py `
  -k "center_frequency or three_requested_figures or manual_parameters" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add -- tools/build_single_waveform_vmd_stvmd_original_notebook.py `
  single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "fix: plot VMD and STVMD center frequencies"
```

### Task 3: Verify Saved Outputs and Full Scientific Regression

**Files:**
- Modify: `tests/test_single_waveform_vmd_stvmd_original_notebook.py`
- Generate: `single_waveform_vmd_stvmd_original.ipynb`

- [ ] **Step 1: Assert exact corrected filenames**

Extend the save test:

```python
assert {path.name for path in tmp_path.glob("*.png")} == {
    "vmd_time_modes.png",
    "vmd_center_frequencies.png",
    "vmd_energy_fraction.png",
    "stvmd_time_modes.png",
    "stvmd_center_frequencies.png",
    "stvmd_energy_fraction.png",
}
```

- [ ] **Step 2: Update the patched notebook parameter cell**

Use:

```python
PLOT_MAX_HZ = None
```

in the end-to-end notebook test.

- [ ] **Step 3: Run the complete corrected notebook test module**

```powershell
python tools/build_single_waveform_vmd_stvmd_original_notebook.py
python -m pytest -q `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
```

Expected: all corrected notebook tests pass.

- [ ] **Step 4: Verify deterministic generation**

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

- [ ] **Step 5: Run all notebook regression tests**

```powershell
python -m pytest -q `
  tests/test_blast_multichannel_stvmd_notebook.py `
  tests/test_single_waveform_stvmd_batched_notebook.py `
  tests/test_single_waveform_vmd_original_notebook.py `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
```

Expected: all tests pass.

- [ ] **Step 6: Inspect the scientific output directly**

Run a diagnostic using `5m.TXT` Tran and assert/report that all returned VMD
center frequencies are present in their corresponding plotted line data,
including centers above 200 Hz.

- [ ] **Step 7: Final diff checks and commit**

```powershell
git diff --check
git status --short
git add -- single_waveform_vmd_stvmd_original.ipynb `
  tests/test_single_waveform_vmd_stvmd_original_notebook.py
git commit -m "test: verify corrected center-frequency analysis"
```
