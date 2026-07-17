from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.compare_5m_long_denoising_methods import (
    align_signals,
    export_results,
    one_sided_amplitude,
)


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
        "time_s",
        "original",
        "ceemdan",
        "vmd_ssa",
        "vmd_mpe_0_60",
    ]
    np.testing.assert_allclose(result["time_s"], time)


def test_align_signals_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number of samples"):
        align_signals(
            np.arange(3.0),
            np.arange(3.0),
            np.arange(2.0),
            np.arange(3.0),
            np.arange(3.0),
        )


def test_one_sided_amplitude_without_window_recovers_sine_amplitude():
    fs = 1024.0
    n = 1024
    time = np.arange(n) / fs
    signal = 2.5 * np.sin(2 * np.pi * 64.0 * time)

    frequency, amplitude = one_sided_amplitude(signal, fs)

    peak = np.argmax(amplitude[1:]) + 1
    assert frequency[peak] == pytest.approx(64.0)
    assert amplitude[peak] == pytest.approx(2.5, rel=1e-12)


def test_export_results_creates_all_artifacts(tmp_path):
    fs = 1000.0
    time = np.arange(1000) / fs
    wave = np.sin(2 * np.pi * 20.0 * time)
    frame = pd.DataFrame(
        {
            "time_s": time,
            "original": wave,
            "ceemdan": wave,
            "vmd_ssa": wave,
            "vmd_mpe_0_60": wave,
        }
    )

    export_results(frame, fs, "mm/s", Path(tmp_path))

    expected = {
        "5m_Long_time_comparison.png",
        "5m_Long_time_comparison.pdf",
        "5m_Long_fft_0_250Hz.png",
        "5m_Long_fft_0_250Hz.pdf",
        "5m_Long_time_fft_comparison.png",
        "5m_Long_time_fft_comparison.pdf",
        "5m_Long_four_signals_aligned.csv",
        "5m_Long_fft_0_250Hz.csv",
        "5m_Long_four_signal_statistics.csv",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert all(path.stat().st_size > 0 for path in tmp_path.iterdir())
