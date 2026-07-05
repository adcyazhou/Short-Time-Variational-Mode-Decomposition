# Single-waveform Original VMD and STVMD Notebook Design

## Goal

Create one independent Jupyter notebook that reads one Instantel waveform,
runs the repository's original VMD first, then runs the repository's original
dynamic STVMD, and shows only each method's modal time histories, modal
amplitude spectra, and modal energy fractions.

The notebook is a new design. It does not import or reuse the previously
created single-waveform notebooks.

## Deliverables

Create:

```text
single_waveform_vmd_stvmd_original.ipynb
tools/build_single_waveform_vmd_stvmd_original_notebook.py
tests/test_single_waveform_vmd_stvmd_original_notebook.py
```

The Python generator makes notebook generation deterministic and extracts the
algorithm source directly from `main_STVMD.ipynb`.

## Source Fidelity

The generator extracts and embeds these three source cells without trimming,
reformatting, or modifying them:

1. `buffer`, `unbuffer`, and `window_norm` helpers;
2. `VMD` class;
3. `STVMD` class.

Tests compare the generated cells with the corresponding source notebook
cells byte for byte. Notebook-specific loading, validation, memory estimation,
analysis adapters, metrics, plotting, and saving are separate cells.

The original STVMD implementation fixes:

```python
self.hop_len = 1
```

The new notebook therefore keeps the hop length at one. It does not expose an
adjustable hop parameter or alter the repository's framing and overlap-add
helpers.

## Notebook Order

The notebook is organized in this order:

1. purpose and dependency notes;
2. imports;
3. one manual parameter cell;
4. a newly written Instantel TXT single-waveform reader;
5. verbatim buffer and overlap-add helpers;
6. verbatim `VMD` class;
7. verbatim `STVMD` class;
8. shared validation, spectrum, energy, plotting, and saving functions;
9. VMD execution and its three figures;
10. dynamic STVMD execution and its three figures;
11. optional PNG and NPZ export.

## Manual Parameters

The only user-editable parameter cell contains:

```python
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
```

`K`, `ALPHA`, `TAU`, `TOL`, and `MAX_ITERS` are shared by VMD and STVMD for a
direct comparison. `VMD_N_FFT` retains its repository meaning as boundary
padding for VMD. `STVMD_WINDOW_LENGTH` is the STVMD transform and window
length.

STVMD uses:

```python
scipy.signal.windows.hamming(
    STVMD_WINDOW_LENGTH, sym=False
)
```

and calls:

```python
model.apply(f_hat, dynamic=True)
```

`MAX_ITERS=1000` matches the original class default. Either method may stop
earlier when its convergence difference falls below `TOL`.

## Input and Processing

The reader loads one selected `Tran`, `Vert`, or `Long` velocity column from
an Instantel TXT file. It extracts sampling rate and pre-trigger duration,
constructs a time axis with the trigger at zero seconds, and retains velocity
in millimetres per second.

The entire waveform is analyzed. There is no external:

- low-pass filtering;
- resampling;
- record cropping;
- length alignment;
- zero padding.

Only padding, short-time framing, windowing, and overlap-add operations already
present in the repository algorithms are used.

The same loaded waveform is supplied independently to VMD and STVMD. Each
adapter returns:

- reconstructed time-domain modes with shape `(K, 1, samples)`;
- the center-frequency output returned by the original algorithm.

The first component is labelled `Residual`. It is the original repository's
zero-centred `k=0` mode, not a post-hoc reconstruction error.

## Spectrum and Energy Definitions

For each reconstructed mode, compute the full-record one-sided FFT amplitude
spectrum. The horizontal axis is frequency in hertz. The vertical axis is
linear velocity amplitude in `mm/s`.

The implementation applies correct one-sided scaling:

- DC is not doubled;
- the Nyquist bin is not doubled for even-length signals;
- all other positive-frequency bins are doubled.

No dB conversion, smoothing, spectral averaging, or additional spectral
window is applied.

For each mode:

\[
E_k = \sum_n u_k[n]^2
\]

and:

\[
p_k = E_k \Big/ \sum_j E_j
\]

Energy fractions include the Residual and sum to one when total modal energy
is nonzero. For an all-zero decomposition, all fractions are zero.

## Figures

Use layout A: separate figures for time, frequency, and energy.

VMD produces, in order:

```text
vmd_time_modes.png
vmd_frequency_modes.png
vmd_energy_fraction.png
```

STVMD then produces:

```text
stvmd_time_modes.png
stvmd_frequency_modes.png
stvmd_energy_fraction.png
```

### Modal time-history figure

- one vertically stacked subplot per component;
- labels `Residual`, `Mode 1`, ..., `Mode K-1`;
- common time axis in seconds;
- velocity axis in `mm/s`;
- a dashed trigger marker at `t=0`.

### Modal frequency figure

- one vertically stacked subplot per component;
- common frequency axis from zero to
  `min(PLOT_MAX_HZ, fs / 2)`;
- linear amplitude axis in `mm/s`;
- no center-frequency curve or time-frequency content.

### Energy fraction figure

- one bar per component, including Residual;
- fractional values displayed above bars;
- no reconstruction-error or residual-error bar.

The notebook does not plot the input waveform, reconstruction diagnostics,
instantaneous frequencies, time-frequency spectra, or comparisons between VMD
and STVMD.

## Saved Outputs

When `SAVE_OUTPUTS=True`, save the six PNG files and:

```text
output/vmd_stvmd_single_waveform/
    vmd_stvmd_single_waveform_results.npz
```

The NPZ includes:

- input path and selected direction;
- sampling rate, input time, and input velocity;
- all VMD modes, modal spectra, modal energies, energy fractions, and returned
  center frequencies;
- all STVMD modes, modal spectra, modal energies, energy fractions, and
  returned dynamic center frequencies;
- all manual parameter values.

When `SAVE_OUTPUTS=False`, figures remain visible in the notebook and no
output directory is created.

## Validation and Resource Reporting

Before execution, validate:

- the input file exists;
- direction is one of `Tran`, `Vert`, or `Long`;
- the loaded signal is a finite one-dimensional array;
- `K >= 2`;
- `ALPHA`, `TAU`, and `TOL` are finite and positive;
- `MAX_ITERS >= 2`;
- `VMD_N_FFT >= 2`;
- `2 <= STVMD_WINDOW_LENGTH <= sample count`;
- `PLOT_MAX_HZ` is finite and positive.

The notebook prints the estimated VMD and STVMD memory requirements and the
STVMD window count before starting either solver. Estimation and warnings do
not alter the algorithms or silently change parameters.

## Verification

Automated tests verify:

- deterministic notebook generation;
- exact equality of all three embedded repository source cells;
- one-source-of-truth manual parameters;
- TXT direction selection, sampling rate, and trigger-aligned time;
- VMD and dynamic STVMD output shapes on a small synthetic record;
- correct single-sided amplitude scaling;
- modal energy fractions sum to one;
- exactly three figures per method with expected content;
- six PNG files and one NPZ are saved when enabled;
- an in-memory notebook with small test parameters executes successfully
  without modifying the committed notebook.
