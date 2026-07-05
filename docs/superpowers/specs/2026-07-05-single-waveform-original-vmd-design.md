# Single-waveform original VMD notebook

## Goal

Create a self-contained Jupyter notebook that reads one waveform from one
Instantel ASCII/TXT file and analyzes the complete waveform with the original
repository VMD implementation.

## Algorithm requirement

- Embed the helper functions and `VMD` class from `main_STVMD.ipynb` verbatim.
- Do not batch, vectorize, or otherwise rewrite the VMD algorithm.
- Use offline whole-record VMD, not sliding-window STVMD.
- Preserve the repository's interpretation of component zero as the
  zero-frequency residual.
- Report one constant center frequency per component; do not present these as
  time-varying instantaneous-frequency tracks.

## Manual configuration

The first parameter cell exposes:

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

`DIRECTION` accepts `Tran`, `Vert`, or `Long`. `N_FFT` retains the original
VMD class parameter name and controls its reflection-padding width; it is not
a short-time window.

## Input

- Parse `Sample Rate` and `Pre-trigger Length` from one Instantel TXT file.
- Parse the three numeric columns `Tran`, `Vert`, and `Long`.
- Select one complete waveform according to `DIRECTION`.
- Do not add CSV conversion, multi-record alignment, or manual time cropping.

## Validation

- Require an existing TXT input and a valid direction.
- Require finite waveform samples and a positive finite sample rate.
- Require integer `K >= 2`, integer `N_FFT >= 2`, integer
  `MAX_ITERS >= 2`, and positive finite `ALPHA`, `TAU`, and `TOL`.

## Outputs

- Input waveform.
- Residual and all oscillatory VMD mode waveforms.
- Constant center-frequency summary for every component.
- Per-component Fourier spectra and frequency-band boundaries.
- Input-versus-reconstruction plot and NRMSE.
- Component energy fractions.
- Optional PNG figures and compressed NumPy results under
  `output/vmd_single_waveform`.

## Files and compatibility

- Add `single_waveform_vmd_original.ipynb`.
- Add a deterministic generator and focused tests.
- Keep all existing notebooks and user TXT files unchanged and untracked.
