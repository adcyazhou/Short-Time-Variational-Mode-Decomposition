# VMD Center-Frequency Correction Design

## Goal

Correct the scientifically misleading VMD presentation in
`single_waveform_vmd_stvmd_original.ipynb`, fix the confirmed VMD convergence
index defect inherited from `main_STVMD.ipynb`, and add numerical regression
tests using a signal with known 20 Hz and 28 Hz components.

## Confirmed causes

The existing `frequency_modes` figure is a full-record FFT amplitude spectrum.
It is valid as a spectrum, but it is not the center-frequency-versus-time plot
the user expects. Every subplot begins at 0 Hz because the plotting code fixes
the spectrum x-axis to `[0, PLOT_MAX_HZ]`.

For `5m.TXT` in the Tran direction, the current VMD returns center frequencies
near 0, 50, 215, and 1378 Hz. The default 200 Hz plot limit therefore hides two
centers and makes their visible spectra appear negligible.

The repository VMD convergence loop also indexes:

```python
u_hat_plus[0, :, i]
```

where `i` is a mode index. This selects a frequency bin. The convergence delta
must instead select the mode dimension:

```python
u_hat_plus[0, :, :, i]
```

The same correction applies to the second iteration buffer.

## Algorithm-source policy

The generator continues to read the VMD class from `main_STVMD.ipynb`. It then
applies exactly one guarded source correction to the confirmed convergence
line. Generation fails if the original line is absent or occurs more than
once. Tests compare the generated VMD source with the original source plus this
single expected replacement, so later accidental algorithm drift is rejected.

The buffer helpers and STVMD class remain byte-for-byte identical to the source
notebook.

## Figure behavior

The three figures for each method remain:

1. modal time histories;
2. modal center frequencies versus time;
3. modal energy fractions.

The FFT amplitude-spectrum figure is removed from displayed and saved figures.
FFT amplitudes remain available in the NPZ because they are useful numerical
results.

For VMD, each scalar center frequency is repeated over the waveform time axis,
producing one horizontal line per component. For dynamic STVMD, each returned
time-varying center-frequency row is plotted against the waveform time axis.
The first component remains labelled `Residual (fixed 0 Hz)`.

The default frequency-axis limit is automatic. No returned center frequency is
hidden. An optional positive manual maximum may be supported only when it is
explicitly set; the default parameter is `PLOT_MAX_HZ = None`.

Saved figure names become:

- `vmd_time_modes.png`
- `vmd_center_frequencies.png`
- `vmd_energy_fraction.png`
- `stvmd_time_modes.png`
- `stvmd_center_frequencies.png`
- `stvmd_energy_fraction.png`

## Scientific regression tests

A deterministic reference signal uses:

```python
fs = 128
t = np.arange(256) / fs
x = np.sin(2*np.pi*20*t) + 0.5*np.sin(2*np.pi*28*t)
```

With `K=3`, `alpha=50`, `tau=1e-5`, `tol=1e-9`, `n_fft=64`, and a sufficient
iteration limit, the two non-residual VMD center frequencies must be within
1 Hz of 20 Hz and 28 Hz. Their modal FFT peaks must also be within 1 Hz of the
known components.

Additional tests verify:

- the VMD center-frequency plot contains constant horizontal lines;
- the dynamic STVMD center-frequency plot accepts and preserves varying rows;
- automatic limits do not hide high-frequency VMD modes;
- exactly six corrected PNG files and the NPZ are saved;
- the generated VMD differs from the source class only at the guarded index
  correction;
- all existing notebook tests remain green.

## Scope

This correction changes only the new combined single-waveform generator,
generated notebook, and its tests. It does not modify `main_STVMD.ipynb`, the
previous VMD/STVMD notebooks, input data, STVMD source, signal preprocessing, or
NPZ numerical arrays.
