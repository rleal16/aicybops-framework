"""
Minimal tests for EVT threshold: fit_baseline, get_current_threshold, and one end-to-end run.
Compares with ads_evt reference when available.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))
from utils.detection import EVTThreshold, SPOT, Status

try:
    from ads_evt.spot import SPOT as RefSPOT
    from ads_evt.spot import dSPOT as RefDSPOT
    ADS_EVT_AVAILABLE = True
except ImportError:
    ADS_EVT_AVAILABLE = False


@pytest.fixture
def baseline_and_stream():
    np.random.seed(123)
    baseline = np.random.normal(0, 1, 200)
    stream = np.random.normal(0, 1, 100)
    risk_level = 1e-3
    init_quantile = 0.98
    return baseline, stream, risk_level, init_quantile


@pytest.mark.skipif(not ADS_EVT_AVAILABLE, reason="ads_evt not available")
def test_fit_baseline(baseline_and_stream):
    """fit_baseline sets initial threshold and peaks consistent with reference."""
    baseline, stream, risk_level, init_quantile = baseline_and_stream
    evt = EVTThreshold(risk_level=risk_level)
    evt.fit_baseline(baseline, init_quantile)
    ref_evt = RefSPOT(q=risk_level)
    ref_evt.fit(init_data=baseline, data=stream)
    ref_evt.initialize(level=init_quantile)
    assert np.isclose(evt._init_threshold, ref_evt._ev._init_threshold, atol=1e-6)
    np.testing.assert_allclose(np.sort(evt._peaks), np.sort(ref_evt._ev._peaks), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not ADS_EVT_AVAILABLE, reason="ads_evt not available")
def test_get_current_threshold(baseline_and_stream):
    """get_current_threshold matches reference extreme_quantile after fit_baseline."""
    baseline, stream, risk_level, init_quantile = baseline_and_stream
    evt = EVTThreshold(risk_level=risk_level)
    evt.fit_baseline(baseline, init_quantile)
    ref_evt = RefSPOT(q=risk_level)
    ref_evt.fit(init_data=baseline, data=stream)
    ref_evt.initialize(level=init_quantile)
    assert np.isclose(evt.get_current_threshold(200), ref_evt._ev.extreme_quantile, atol=1e-6)


@pytest.mark.skipif(not ADS_EVT_AVAILABLE, reason="ads_evt not available")
def test_evt_end_to_end(baseline_and_stream):
    """SPOT fit_baseline + run_stream matches reference alarms and thresholds."""
    baseline, stream, risk_level, init_quantile = baseline_and_stream
    spot = SPOT(risk_level=risk_level)
    spot.fit_baseline(baseline, stream, init_quantile)
    ref_spot = RefSPOT(q=risk_level)
    ref_spot.fit(init_data=baseline, data=stream)
    ref_spot.initialize(level=init_quantile)
    ours = spot.run_stream()
    ref_result = ref_spot.run()
    assert ours["alarms"] == ref_result["alarms"]
    np.testing.assert_allclose(ours["thresholds"], ref_result["thresholds"], rtol=1e-2, atol=1e-2)
