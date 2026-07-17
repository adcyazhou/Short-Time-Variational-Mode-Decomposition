from tools.analyze_5m_long_mpe_denoising import format_threshold


def test_format_threshold_preserves_two_decimal_threshold():
    assert format_threshold(0.55) == "0.55"

