"""
Tests for session_log_to_labels and report_unlabellable_periods.

Labels are built from metrics timestamps (1:1); 0=normal, 1=anomaly, -1=unlabellable.
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from data.session_log_to_labels import session_log_to_labels, report_unlabellable_periods


def _metrics_df_from_timestamps(seconds_utc: list) -> pd.DataFrame:
    """Build a minimal metrics DataFrame with _time (unix ms at second boundary)."""
    index = pd.to_datetime(seconds_utc, unit="s", utc=True).floor("s")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    time_ms = (index.astype("int64") // 10**6).astype("int64")
    return pd.DataFrame({"_time": time_ms})


def test_session_log_to_labels_requires_metrics(tmp_path):
    """Returns None when metrics_df is None or empty or missing _time."""
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:05.000Z",
        "fault_events": [],
        "normal_windows": [{"start_ts": "2026-03-04T14:00:00.000Z", "end_ts": "2026-03-04T14:00:05.000Z", "label": "normal"}],
    }
    out = tmp_path / "labels.csv"
    assert session_log_to_labels(session_log, str(out), metrics_df=None) is None
    assert session_log_to_labels(session_log, str(out), metrics_df=pd.DataFrame()) is None
    assert session_log_to_labels(session_log, str(out), metrics_df=pd.DataFrame({"x": [1]})) is None
    assert not out.exists()


def test_session_log_to_labels_metrics_driven_0_1(tmp_path):
    """Labels are 1:1 with metrics timestamps; fault -> 1, normal -> 0 (point-in-interval, inclusive)."""
    # Metrics at 14:00:00, 14:00:01, 14:00:02 (three seconds)
    # Fault [14:00:01, 14:00:02]; 14:00:02 is inside fault (inclusive end) -> 1
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    seconds = [int(base_ts.timestamp()) + i for i in range(3)]
    metrics_df = _metrics_df_from_timestamps(seconds)
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:03.000Z",
        "fault_events": [{"start_ts": "2026-03-04T14:00:01.000Z", "end_ts": "2026-03-04T14:00:02.000Z"}],
        "normal_windows": [
            {"start_ts": "2026-03-04T14:00:00.000Z", "end_ts": "2026-03-04T14:00:01.000Z", "label": "normal"},
            {"start_ts": "2026-03-04T14:00:02.000Z", "end_ts": "2026-03-04T14:00:03.000Z", "label": "normal"},
        ],
    }
    out = tmp_path / "labels.csv"
    result = session_log_to_labels(session_log, str(out), metrics_df=metrics_df)
    assert result == str(out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["_time", "anomaly_label"]
    assert len(df) == 3
    assert df["anomaly_label"].tolist() == [0, 1, 1]


def test_session_log_to_labels_unlabellable_minus_one(tmp_path):
    """Timestamps outside all session_log periods get -1 (point-in-interval, inclusive)."""
    # Metrics at 14:00:00 .. 14:00:04; normal window [14:00:01, 14:00:03] (inclusive)
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    seconds = [int(base_ts.timestamp()) + i for i in range(5)]
    metrics_df = _metrics_df_from_timestamps(seconds)
    session_log = {
        "session_start": "2026-03-04T14:00:01.000Z",
        "session_end": "2026-03-04T14:00:03.000Z",
        "fault_events": [],
        "normal_windows": [{"start_ts": "2026-03-04T14:00:01.000Z", "end_ts": "2026-03-04T14:00:03.000Z", "label": "normal"}],
    }
    out = tmp_path / "labels.csv"
    session_log_to_labels(session_log, str(out), metrics_df=metrics_df)
    df = pd.read_csv(out)
    assert len(df) == 5
    # 14:00:00 outside -> -1; 14:00:01, 14:00:02, 14:00:03 in [14:00:01, 14:00:03] -> 0; 14:00:04 outside -> -1
    assert df["anomaly_label"].tolist() == [-1, 0, 0, 0, -1]


def test_report_unlabellable_periods_empty(tmp_path):
    """When no -1 rows, returns empty list."""
    df = pd.DataFrame({"_time": [1709560800000, 1709560801000], "anomaly_label": [0, 1]})
    df.to_csv(tmp_path / "labels.csv", index=False)
    periods = report_unlabellable_periods(str(tmp_path / "labels.csv"))
    assert periods == []


def test_report_unlabellable_periods_contiguous(tmp_path):
    """Consecutive -1 timestamps are merged into one period."""
    # 14:00:00, 14:00:01, 14:00:02 all -1 -> one period
    df = pd.DataFrame({
        "_time": [1709560800000, 1709560801000, 1709560802000, 1709560803000],
        "anomaly_label": [-1, -1, -1, 0],
    })
    df.to_csv(tmp_path / "labels.csv", index=False)
    periods = report_unlabellable_periods(str(tmp_path / "labels.csv"))
    assert len(periods) == 1
    start, end = periods[0]
    # 1709560800000 ms = 2024-03-04 14:00:00 UTC
    assert start == pd.Timestamp("2024-03-04 14:00:00")
    assert end == pd.Timestamp("2024-03-04 14:00:02")


def test_report_unlabellable_periods_nonexistent_path():
    """Returns empty list when file does not exist."""
    assert report_unlabellable_periods("/nonexistent/labels.csv") == []


def test_session_log_to_labels_raises_when_normal_windows_missing(tmp_path):
    """Raises ValueError when session_log has no normal_windows (required from API)."""
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    metrics_df = _metrics_df_from_timestamps([int(base_ts.timestamp())])
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:01.000Z",
        "fault_events": [],
        # no normal_windows
    }
    out = tmp_path / "labels.csv"
    with pytest.raises(ValueError, match="normal_windows"):
        session_log_to_labels(session_log, str(out), metrics_df=metrics_df)


def test_session_log_to_labels_raises_when_normal_windows_not_list(tmp_path):
    """Raises ValueError when session_log['normal_windows'] is not a list."""
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    metrics_df = _metrics_df_from_timestamps([int(base_ts.timestamp())])
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:01.000Z",
        "fault_events": [],
        "normal_windows": "not a list",
    }
    out = tmp_path / "labels.csv"
    with pytest.raises(ValueError, match="must be a list"):
        session_log_to_labels(session_log, str(out), metrics_df=metrics_df)


def test_overlap_boundaries_fault_and_normal(tmp_path):
    """Point-in-interval: t in [start_ts, end_ts] (inclusive). Fault overrides normal."""
    # Session 14:00:00–14:00:03; fault [14:00:01, 14:00:02]; normal [14:00:00, 14:00:01] and [14:00:02, 14:00:03].
    # Metrics at 14:00:00, 14:00:01, 14:00:02.
    # 14:00:00 in normal [14:00:00, 14:00:01] -> 0.
    # 14:00:01 in fault [14:00:01, 14:00:02] -> 1.
    # 14:00:02 in fault [14:00:01, 14:00:02] (inclusive end) -> 1 (fault overrides).
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    seconds = [int(base_ts.timestamp()) + i for i in range(3)]
    metrics_df = _metrics_df_from_timestamps(seconds)
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:03.000Z",
        "fault_events": [{"start_ts": "2026-03-04T14:00:01.000Z", "end_ts": "2026-03-04T14:00:02.000Z"}],
        "normal_windows": [
            {"start_ts": "2026-03-04T14:00:00.000Z", "end_ts": "2026-03-04T14:00:01.000Z", "label": "normal"},
            {"start_ts": "2026-03-04T14:00:02.000Z", "end_ts": "2026-03-04T14:00:03.000Z", "label": "normal"},
        ],
    }
    out = tmp_path / "labels.csv"
    session_log_to_labels(session_log, str(out), metrics_df=metrics_df)
    df = pd.read_csv(out)
    assert len(df) == 3
    assert df["anomaly_label"].tolist() == [0, 1, 1]


def test_overlap_inclusive_interval_end(tmp_path):
    """Point-in-interval: t in [start_ts, end_ts] (inclusive). 14:00:01 is inside [14:00:00, 14:00:01]."""
    # Normal window [14:00:00, 14:00:01]. Metrics at 14:00:00 and 14:00:01.
    # 14:00:00 in [14:00:00, 14:00:01] -> 0.
    # 14:00:01 in [14:00:00, 14:00:01] (inclusive end) -> 0.
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    seconds = [int(base_ts.timestamp()), int(base_ts.timestamp()) + 1]
    metrics_df = _metrics_df_from_timestamps(seconds)
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:01.000Z",
        "fault_events": [],
        "normal_windows": [{"start_ts": "2026-03-04T14:00:00.000Z", "end_ts": "2026-03-04T14:00:01.000Z", "label": "normal"}],
    }
    out = tmp_path / "labels.csv"
    session_log_to_labels(session_log, str(out), metrics_df=metrics_df)
    df = pd.read_csv(out)
    assert len(df) == 2
    assert df["anomaly_label"].tolist() == [0, 0]


def test_labels_align_1_to_1_with_metrics(tmp_path):
    """Output has one row per metric row (1:1); _time values match metrics exactly."""
    base_ts = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").tz_localize(None)
    seconds = [int(base_ts.timestamp()) + i for i in range(5)]
    metrics_df = _metrics_df_from_timestamps(seconds)
    # Duplicate one second to simulate multiple rows per second
    extra = pd.DataFrame({"_time": [metrics_df["_time"].iloc[2]]})
    metrics_with_dup = pd.concat([metrics_df, extra], ignore_index=True)
    session_log = {
        "session_start": "2026-03-04T14:00:00.000Z",
        "session_end": "2026-03-04T14:00:05.000Z",
        "fault_events": [],
        "normal_windows": [{"start_ts": "2026-03-04T14:00:00.000Z", "end_ts": "2026-03-04T14:00:05.000Z", "label": "normal"}],
    }
    out = tmp_path / "labels.csv"
    session_log_to_labels(session_log, str(out), metrics_df=metrics_with_dup)
    df = pd.read_csv(out)
    # One label per metric row (6 rows)
    assert len(df) == len(metrics_with_dup)
    pd.testing.assert_series_equal(
        df["_time"].astype(int),
        metrics_with_dup["_time"].astype(int),
        check_names=False,
    )
