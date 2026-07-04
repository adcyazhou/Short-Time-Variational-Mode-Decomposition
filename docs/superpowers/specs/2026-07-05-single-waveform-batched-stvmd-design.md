# Single-waveform batched STVMD notebook

## Goal

Create a self-contained Jupyter notebook for applying the repository's
batch-optimized dynamic STVMD implementation to one waveform selected from one
Instantel ASCII/TXT file.

## User workflow

The first configuration cell exposes only manual settings:

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

The user edits the values, runs the notebook from top to bottom, and receives
the analysis for the complete selected waveform. The notebook does not provide
an additional start/end-time crop.

## Input and validation

- Parse one Instantel ASCII/TXT file and its `Sample Rate` and
  `Pre-trigger Length` metadata.
- Allow `DIRECTION` to be `Tran`, `Vert`, or `Long`.
- Extract the selected direction as a single-channel array.
- Require positive finite sampling rate and algorithm parameters.
- Require `K >= 2`, integer `WINDOW_LENGTH >= 2`, integer
  `BATCH_WINDOWS >= 1`, and a waveform at least as long as the window.
- Accept any valid integer window length rather than limiting it to the
  repository demonstration tuple.

## Algorithm

- Embed the batch-optimized dynamic STVMD implementation so the notebook is
  independent of the other notebooks.
- Keep the sliding hop at one sample.
- Process all sliding windows in groups controlled by `BATCH_WINDOWS`.
- Treat component zero as the residual and components `1..K-1` as oscillatory
  modes.

## Outputs

- Input waveform and mean time-frequency power.
- One plot per residual/mode waveform.
- Instantaneous center-frequency tracks for modes `1..K-1`.
- Input-versus-reconstruction plot and NRMSE.
- Mode energy fractions.
- Fourier spectrum beside the time-frequency spectrum with mode tracks and
  frequency-band boundaries.
- Optional PNG and compressed NumPy result output under a dedicated
  `output/stvmd_single_waveform` directory.

## Files and compatibility

- Add `single_waveform_stvmd_batched.ipynb`.
- Add or update focused tests for notebook structure and a small quick-mode
  execution.
- Keep `blast_multichannel_stvmd.ipynb` and
  `blast_multichannel_stvmd_batched.ipynb` unchanged.
- Do not stage or modify the user's `5m.TXT`, `10m.TXT`, or `15m.TXT` files.
