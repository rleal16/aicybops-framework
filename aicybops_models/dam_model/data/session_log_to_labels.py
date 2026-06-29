"""
Build anomaly_labels.csv from session_log and collected metrics.

Uses metrics timestamps as-is (no synthetic grid).
Per row: anomaly=1, normal=0, or -1 when outside all session windows.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def _parse_iso(ts: str) -> pd.Timestamp:
    """Parse ISO timestamp; normalize to UTC for consistent comparison."""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC")
    return t


def _build_interval_list(session_log: Dict[str, Any]) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Build a list of (start_ts, end_ts, label) from session_log.
    Label: 0 = normal, 1 = anomaly.
    Normal windows are added first; anomalies override on overlap.
    """
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for w in session_log.get("normal_windows") or []:
        s, e = w.get("start_ts"), w.get("end_ts")
        if not s or not e:
            continue
        try:
            intervals.append((_parse_iso(s), _parse_iso(e), 0))
        except Exception:
            continue
    for ev in session_log.get("fault_events") or []:
        s, e = ev.get("start_ts"), ev.get("end_ts")
        if not s or not e:
            continue
        try:
            intervals.append((_parse_iso(s), _parse_iso(e), 1))
        except Exception:
            continue
    return intervals


def _label_for_timestamp(t: pd.Timestamp, intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]]) -> int:
    """Return 1 if t falls in any anomaly interval, 0 if in any normal interval, else -1."""
    if not intervals:
        return -1
    if t.tz is None:
        t = t.tz_localize("UTC")
    elif t.tz != intervals[0][0].tz:
        t = t.tz_convert(intervals[0][0].tz)
    label = -1
    for start_ts, end_ts, lab in intervals:
        if start_ts <= t <= end_ts:
            if lab == 1:
                return 1
            label = 0
    return label


def _warn_fallback(msg: str) -> None:
    print(
        f"[session_log_to_labels] {msg} Treating all points as normal (0) so training can proceed.",
        file=sys.stderr,
    )


def session_log_to_labels(
    session_log: Dict[str, Any],
    output_path: str,
    metrics_df: pd.DataFrame,
) -> Optional[str]:
    """
    Build labels from session_log with a 1:1 match to metrics rows/timestamps.

    Uses metrics timestamps as-is and assigns anomaly=1, normal=0, or -1.

    Args:
        session_log: Dict with fault_events and normal_windows (each with start_ts, end_ts).
        output_path: Path to write anomaly_labels.csv.
        metrics_df: Collected metrics DataFrame with _time column. Required.

    Returns:
        output_path if successful, None on error.
    """
    if metrics_df is None or metrics_df.empty or "_time" not in metrics_df.columns:
        return None

    if "normal_windows" not in session_log:
        raise ValueError(
            "session_log must contain 'normal_windows' (list). "
            "Ensure session log is from Testbed/collect_metrics API."
        )
    nw = session_log["normal_windows"]
    if not isinstance(nw, list):
        raise ValueError(
            "session_log['normal_windows'] must be a list. "
            f"Got {type(nw).__name__}. Ensure session log is from Testbed/collect_metrics API."
        )

    intervals = _build_interval_list(session_log)
    # Use metrics timestamps as-is.
    ts = pd.to_datetime(metrics_df["_time"], unit="ms")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    if not intervals:
        # No session windows.
        labels = pd.Series(0, index=ts.index)
        _warn_fallback("No session windows available (empty).")
    else:
        labels = ts.apply(lambda t: _label_for_timestamp(t, intervals))
        if labels.eq(-1).all():
            # Session windows exist but do not overlap metrics.
            print(
                "[session_log_to_labels] Session time range does not overlap metrics — "
                "labels not generated. Collect metrics for the session time window "
                "(use_session_time_range=True) or provide a matching session log.",
                file=sys.stderr,
            )
            return None

    # Preserve metrics _time values for 1:1 alignment.
    time_ms = metrics_df["_time"].values
    df = pd.DataFrame({"_time": time_ms, "anomaly_label": labels.values.astype(int)})
    # Long-format metrics produce many rows per _time (one per measurement/field);
    # collapse to one label per timestamp so downstream reindex/align stays unique.
    # All rows for the same _time share the same label by construction
    # (label is computed from the timestamp), so keeping the first is safe.
    df = df.drop_duplicates(subset="_time", keep="first").sort_values("_time")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def report_unlabellable_periods(label_path: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find timestamps with anomaly_label == -1 and merge into contiguous periods.

    Args:
        label_path: Path to anomaly_labels.csv (columns _time, anomaly_label).

    Returns:
        List of (start_ts, end_ts) for each contiguous unlabellable period.
    """
    path = Path(label_path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "_time" not in df.columns or "anomaly_label" not in df.columns:
        return []
    unlabelled = df[df["anomaly_label"] == -1].copy()
    if unlabelled.empty:
        return []
    unlabelled["_time"] = pd.to_datetime(unlabelled["_time"], unit="ms")
    unlabelled = unlabelled.sort_values("_time")
    timestamps = unlabelled["_time"].values
    one_sec = pd.Timedelta("1s")
    periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    period_start = pd.Timestamp(timestamps[0])
    period_end = period_start + one_sec
    for i in range(1, len(timestamps)):
        t = pd.Timestamp(timestamps[i])
        if t <= period_end:
            period_end = t + one_sec
        else:
            periods.append((period_start, period_end - one_sec))
            period_start = t
            period_end = t + one_sec
    periods.append((period_start, period_end - one_sec))
    return periods
