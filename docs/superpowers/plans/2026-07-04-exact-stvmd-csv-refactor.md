# Exact STVMD CSV Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom batched STVMD implementation with the repository’s verbatim Numba/tqdm STVMD source and make pandas-generated CSV files the analysis input.

**Architecture:** The deterministic Notebook builder extracts cells 0, 1, and 3 from `main_STVMD.ipynb` and embeds their source unchanged. Separate Notebook cells handle pandas TXT-to-CSV conversion, common-length channel assembly, result adaptation, figures, and exports without modifying the original algorithm.

**Tech Stack:** Python, Jupyter/nbformat/nbclient, pandas, NumPy, SciPy, Numba, tqdm, Matplotlib, pytest.

---

### Task 1: Add pandas ASCII-to-CSV conversion

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`
- Modify: `.gitignore`

- [ ] Write failing tests that call `convert_instantel_ascii_to_csv()` on a synthetic Instantel file and assert:
  - returned metadata contains `fs` and `pretrigger_seconds`;
  - `data_csv/sample.csv` exists;
  - pandas reads columns `Sample, Time_s, Tran, Vert, Long`;
  - time begins at `-0.5`;
  - velocity values and row count are unchanged.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q` and confirm the function is missing.
- [ ] Implement pandas parsing:

```python
def convert_instantel_ascii_to_csv(txt_path, csv_path):
    txt_path, csv_path = Path(txt_path), Path(csv_path)
    lines = txt_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
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
        raise ValueError(f"{txt_path.name}: 未找到 Tran/Vert/Long 表头")
    fs = _metadata_number(metadata, "Sample Rate")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    frame = pd.read_csv(
        txt_path, sep=r"\s+", skiprows=header_index + 1,
        names=["Tran", "Vert", "Long"], engine="python",
    )
    if frame.shape[1] != 3 or frame.isna().any().any():
        raise ValueError(f"{txt_path.name}: 数值区不是有效的三列数据")
    frame.insert(0, "Time_s", np.arange(len(frame)) / fs - pretrigger)
    frame.insert(0, "Sample", np.arange(len(frame), dtype=int))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return {"fs": fs, "pretrigger_seconds": pretrigger, "metadata": metadata}
```

- [ ] Add `data_csv/` to `.gitignore`.
- [ ] Regenerate Notebook and verify tests pass.
- [ ] Commit: `feat: add pandas ASCII to CSV conversion`.

### Task 2: Embed the repository STVMD source verbatim

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] Write failing tests that:
  - read source cells 0, 1, and 3 from `main_STVMD.ipynb`;
  - locate generated cells tagged `original-stvmd`;
  - compare their source strings exactly and in order;
  - assert `@jit(nopython=True, cache=True)`, `self.hop_len = 1`, `tqdm`, and `apply_dynamic` exist;
  - assert `run_dynamic_stvmd_batched` no longer exists.
- [ ] Run tests and confirm they fail against the current batched Notebook.
- [ ] In the builder, load the source Notebook:

```python
source_nb = nbformat.read(ROOT / "main_STVMD.ipynb", as_version=4)
original_cells = [
    new_code_cell(
        source_nb.cells[index].source,
        metadata={"tags": ["core", "original-stvmd"]},
    )
    for index in (0, 1, 3)
]
```

- [ ] Remove the custom `STVMD` string, `_solve_dynamic_batch`, and `run_dynamic_stvmd_batched` from the builder.
- [ ] Insert the three original cells before all adapter functions.
- [ ] Regenerate and run source-parity tests.
- [ ] Commit: `refactor: use repository STVMD implementation`.

### Task 3: Read CSVs and adapt original STVMD outputs

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] Write failing tests for:
  - `load_csv_direction_inputs()` reading 5m/10m/15m in that order;
  - truncation to the shortest CSV;
  - output shapes `(3, common_n)` for Tran, Vert, Long;
  - `run_original_stvmd()` returning modes `(K, 3, N)`, center frequencies `(K, N)`, mean TF power, iterations/convergence metadata.
- [ ] Implement:

```python
def load_csv_direction_inputs(csv_paths, fs, pretrigger_seconds):
    order = ("5m", "10m", "15m")
    frames = {key: pd.read_csv(csv_paths[key]) for key in order}
    common_n = min(len(frame) for frame in frames.values())
    signals = {
        direction: np.vstack([
            frames[key][direction].to_numpy(float)[:common_n] for key in order
        ])
        for direction in ("Tran", "Vert", "Long")
    }
    time_s = frames["5m"]["Time_s"].to_numpy(float)[:common_n]
    return frames, signals, time_s


def run_original_stvmd(x, fs, K, alpha, window_length, tau, tol, max_iters):
    window = scipy.signal.windows.hamming(window_length, sym=False)
    model = STVMD(
        num_channel=x.shape[0], window_func=window, alpha=alpha,
        n_fft=window_length, K=K, tol=tol, tau=tau,
        maxiters=max_iters,
    )
    f_hat_s, windowed = model.prepare_offline(x)
    u_hat, omega = model.apply(f_hat_s, dynamic=True)
    modes = model.postprocess(u_hat)
    return {
        "modes": modes,
        "u_hat": u_hat,
        "center_freq_hz": omega * (fs / 2.0),
        "mean_tf_power": np.mean(np.abs(f_hat_s) ** 2, axis=0),
        "windowed_signal": windowed,
    }
```

- [ ] Adapt `summarize_stvmd_result()` to accept the original result dictionary.
- [ ] Add an external memory estimate function; do not edit original source.
- [ ] Run tests using a short synthetic three-channel signal, `K=3`, window16, and small `max_iters`.
- [ ] Commit: `feat: adapt original STVMD to CSV inputs`.

### Task 4: Rebuild the detailed Notebook workflow

**Files:**
- Modify: `tools/build_blast_multichannel_stvmd_notebook.py`
- Regenerate: `blast_multichannel_stvmd.ipynb`
- Modify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] Update configuration and Markdown to explain:
  - pandas CSV generation;
  - exact-source STVMD cells;
  - Numba’s actual scope;
  - tqdm progress;
  - no batching and expected memory/runtime cost.
- [ ] Add conversion cell that writes all three CSVs, then a separate cell that reads those CSVs.
- [ ] Replace all calls to the batched function with `run_original_stvmd`.
- [ ] Retain explicit Tran → Vert → Long sections and the four approved figure families.
- [ ] Keep `STVMD_QUICK_TEST=1`, but use a short signal and low `MAX_ITERS`; do not alter original class source.
- [ ] Update save cells to write PNGs and `stvmd_results.npz`.
- [ ] Update smoke test to assert CSV files are produced in a temporary/quick-test directory and all Notebook cells execute.
- [ ] Run all tests and visually inspect the spectrum/IF mapping quick figure.
- [ ] Commit: `refactor: rebuild notebook around exact STVMD`.

### Task 5: Final verification and integration

**Files:**
- Verify: `blast_multichannel_stvmd.ipynb`
- Verify: `main_STVMD.ipynb`
- Verify: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Run the builder twice and compare SHA256 hashes.
- [ ] Confirm exact source-cell equality programmatically.
- [ ] Convert the real ASCII files and verify CSV row counts 33493, 14336, and 23176.
- [ ] Verify common direction shapes are `(3, 14336)`.
- [ ] Run a short real-data slice through original `STVMD` with K=3, window16, and low max iterations.
- [ ] Run `git diff --check` and confirm only source TXT files remain untracked.
- [ ] Do not claim the complete full-data decomposition was executed.
