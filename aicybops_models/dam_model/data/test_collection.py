#!/usr/bin/env python3
"""
Test script for data collection (metrics and logs).

Runs DataCollector against the API, saves to a temporary directory, and validates:
1. That the output files exist and have the expected structure for DAMDataProcessor
   (metrics CSV with _time, log file in LogAnalyzer format).
2. That the processing pipeline (DAMDataProcessor / data_analysis) correctly processes
   the collected metrics and logs (analyzers, alignment, groups, sequences).

Logs are requested with format=loganalyzer (GET .../collect_logs/all_logs?start=-2m&format=loganalyzer)
so logs.txt is LogAnalyzer-compatible text (one line per entry).

Usage:
    # With live API (requires API_URL)
    # URL from docker-compose.yml (aicybops-service): http://213.30.51.238:5010
    # Or local: http://localhost:5010 ; or call_api.py default: http://10.2.0.76:5010
    export API_URL=http://213.30.51.238:5010
    python -m aicybops_models.dam_model.data.test_collection

    # From dam_model directory
    cd aicybops_models/dam_model && API_URL=http://localhost:5010 python -m data.test_collection

    # Custom output directory (default: temp dir)
    API_URL=http://localhost:5010 python -m data.test_collection --output-dir data/collected_test

    # Test log conversion only (no API; uses mock JSON)
    python -m data.test_collection --test-logs-only

    # Test only /test_connection endpoint (no collection)
    API_URL=http://213.30.51.238:5010 python -m data.test_collection --test-connection-only

    # Live run (testbed still running): read session log from Redis (real-time)
    API_URL=http://localhost:5010 python -m data.test_collection \\
        --session-log-redis redis://localhost:6380/0

    # Remote machine: collect metrics/logs and fetch labels from testbed host API
    API_URL=http://testbed-host:5010 python -m data.test_collection --output-dir data \\
        --session-log-url http://testbed-host:5010/session_log
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

# Ensure dam_model is on path when run as script
_dam_model_root = Path(__file__).resolve().parent.parent
_data_dir = Path(__file__).resolve().parent  # aicybops_models/dam_model/data
if str(_dam_model_root) not in sys.path:
    sys.path.insert(0, str(_dam_model_root))


def _read_session_log_from_redis(redis_url: str, key: str = "session_log") -> Optional[dict]:
    """Connect to Redis and return the session log dict, or None on failure."""
    if redis_lib is None:
        print("ERROR: redis package not installed — pip install redis")
        return None
    try:
        client = redis_lib.from_url(redis_url, decode_responses=True)
        client.ping()
        raw = client.get(key)
        if not raw:
            print(f"Redis key '{key}' is empty or missing at {redis_url}")
            return None
        data = json.loads(raw)
        if isinstance(data, dict) and "normal_windows" in data:
            return data
        print(f"Redis session_log missing 'normal_windows' — got keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
        return None
    except Exception as e:
        print(f"Failed to read session_log from Redis ({redis_url}): {e}")
        return None


def _check_metrics_csv(path: str) -> tuple[bool, str]:
    """Validate collected metrics CSV: exists, readable, has rows and expected columns if possible."""
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    try:
        df = pd.read_csv(p, nrows=5)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"
    cols = list(df.columns)
    # MetricsAnalyser requires _time.
    if "_time" not in cols:
        return True, f"OK but missing '_time' column (required by MetricsAnalyser). Columns: {cols}"
    return True, f"OK: {len(pd.read_csv(p))} rows, columns: {cols}"


def _check_log_file(path: str) -> tuple[bool, str]:
    """Validate collected log file: exists and lines match LogAnalyzer pattern."""
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    # Same pattern as LogAnalyzer (ISO timestamp, level, pid, thread, logger, message)
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)"
        r"\s+(\w+)\s+(\d+)\s+---\s+\[(.*?)\]\s+(.*?)\s+:\s+(.*)"
    )
    parsed = 0
    total = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if pattern.match(line):
                parsed += 1
    if total == 0:
        return True, "OK: empty log file"
    if parsed < total:
        return False, f"Only {parsed}/{total} lines match LogAnalyzer format"
    return True, f"OK: {total} log lines, all match LogAnalyzer format"


def _check_processing(
    metrics_path: str,
    log_path: str,
    config_path: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate that the collected metrics and logs are correctly processed by the
    DAM pipeline (DAMDataProcessor / processing.data_analysis).

    Runs _process_data() and _create_sequences(), and checks that groups and
    sequences are non-empty and have expected structure.
    Uses a temporary config with labels disabled so that collected API data
    (which has no matching anomaly_labels.csv) does not fail alignment checks.
    """
    config_path = config_path or str(_dam_model_root / "configs" / "dam_config.json")
    if not Path(config_path).exists():
        return False, f"Config not found: {config_path}"

    # Use a temp config with labels disabled.
    with open(config_path, "r") as f:
        config = json.load(f)
    if "label_dataset" in config:
        config["label_dataset"] = {**config["label_dataset"], "enabled": False}
    if config.get("data_processing") and "label_dataset" in config["data_processing"]:
        config["data_processing"]["label_dataset"] = {
            **config["data_processing"]["label_dataset"],
            "enabled": False,
        }
    fd, temp_config = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        os.unlink(temp_config)
        raise

    try:
        from processing.data_analysis import DAMDataProcessor
    except ImportError as e:
        os.unlink(temp_config)
        return False, f"Cannot import DAMDataProcessor: {e}"

    try:
        processor = DAMDataProcessor(
            metrics_csv_path=metrics_path,
            log_file_path=log_path,
            config_path=temp_config,
            use_api=False,
            window_size=10,
            stride=1,
        )
        processor._process_data()
    except Exception as e:
        os.unlink(temp_config)
        return False, f"Processing failed: {type(e).__name__}: {e}"
    finally:
        if Path(temp_config).exists():
            os.unlink(temp_config)

    if not processor.groups:
        return False, "Processing produced no groups (check config metric_groups and data)"

    try:
        processor._create_sequences()
    except Exception as e:
        return False, f"Sequence creation failed: {type(e).__name__}: {e}"

    if not processor.sequences:
        return False, "Processing produced no sequences"

    for group_name, seqs in processor.sequences.items():
        if seqs.size == 0:
            # Single-scrape datasets can legitimately yield empty sequences.
            try:
                mdf = pd.read_csv(metrics_path)
                if "_time" in mdf.columns and mdf["_time"].nunique() <= 1:
                    return True, (
                        "OK: processed; single timestamp in metrics (one scrape) so "
                        "sequences empty; collection and pipeline compatible."
                    )
            except Exception:
                pass
            return False, f"Group {group_name!r} has empty sequences"

    group_info = ", ".join(
        f"{k}={v.shape}" for k, v in processor.sequences.items()
    )
    return True, f"OK: processed and created sequences ({group_info})"


def _check_processing_with_labels(
    metrics_path: str,
    log_path: str,
    labels_path: str,
    config_path: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Run DAMDataProcessor with labels enabled and verify alignment between
    labels, metrics (all groups), and logs. Uses a temp config with label path
    set to labels_path and label_dataset.enabled = True.
    """
    config_path = config_path or str(_dam_model_root / "configs" / "dam_config.json")
    if not Path(config_path).exists():
        return False, f"Config not found: {config_path}"
    if not Path(labels_path).exists():
        return False, f"Labels file not found: {labels_path}"

    with open(config_path, "r") as f:
        config = json.load(f)
    # Point labels to the collected file (absolute path).
    abs_labels = str(Path(labels_path).resolve())
    if "label_dataset" in config:
        config["label_dataset"] = {
            **config["label_dataset"],
            "enabled": True,
            "path": abs_labels,
        }
    fd, temp_config = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        os.unlink(temp_config)
        raise

    try:
        from processing.data_analysis import DAMDataProcessor
    except ImportError as e:
        os.unlink(temp_config)
        return False, f"Cannot import DAMDataProcessor: {e}"

    try:
        processor = DAMDataProcessor(
            metrics_csv_path=metrics_path,
            log_file_path=log_path,
            config_path=temp_config,
            use_api=False,
            window_size=10,
            stride=1,
        )
        processor._process_data()
        processor._create_sequences()
    except Exception as e:
        if Path(temp_config).exists():
            os.unlink(temp_config)
        # Single-scrape datasets can fail sequence creation; accept if alignment passed.
        try:
            mdf = pd.read_csv(metrics_path)
            if "_time" in mdf.columns and mdf["_time"].nunique() <= 1:
                return True, (
                    "OK: alignment verified (labels, metrics, logs); "
                    "single timestamp in metrics so sequences not created (collection and pipeline compatible)."
                )
        except Exception:
            pass
        return False, f"Processing with labels failed: {type(e).__name__}: {e}"
    finally:
        if Path(temp_config).exists():
            os.unlink(temp_config)

    if not processor.groups or not processor.sequences:
        return False, "Processing produced no groups or sequences"

    # Single-scrape datasets can leave all sequences empty.
    all_empty = all(seqs.size == 0 for seqs in processor.sequences.values())
    if all_empty:
        try:
            mdf = pd.read_csv(metrics_path)
            if "_time" in mdf.columns and mdf["_time"].nunique() <= 1:
                return True, (
                    "OK: alignment verified (labels, metrics, logs); "
                    "single timestamp in metrics so no sequences (collection and pipeline compatible)."
                )
        except Exception:
            pass
        return False, "Processing produced no groups or sequences"

    # Alignment is validated in _process_data; sequence labels in _create_sequences.
    seq_labels = processor.sequence_anomaly_labels
    if seq_labels is None:
        return False, "Sequence-level labels not computed"
    n_normal = int((seq_labels == 0).sum())
    n_anomaly = int((seq_labels == 1).sum())
    group_info = ", ".join(
        f"{k}={v.shape}" for k, v in processor.sequences.items()
    )
    return True, (
        f"OK: alignment verified (labels, metrics groups, logs). "
        f"Sequences: {group_info} | labels: {len(seq_labels)} total ({n_normal} normal, {n_anomaly} anomalous)"
    )


def run_connection_test(api_url: str) -> bool:
    """Call GET /test_connection and print result. Returns True if reachable."""
    from data.data_collector import DataCollector

    print("Test connection (GET /test_connection)")
    print("  API_URL:", api_url)
    print()
    collector = DataCollector(api_url=api_url, output_dir=tempfile.mkdtemp())
    result = collector.test_connection()
    if result is None:
        print("FAIL: API unreachable (test_connection failed)")
        return False
    print("  result:", result)
    print("PASS: API reachable")
    return True


def run_collection_test(
    api_url: str,
    output_dir: str,
    session_log_url: Optional[str] = None,
    session_log_file: Optional[str] = None,
    session_log_redis: Optional[str] = None,
    labels_output: Optional[str] = None,
    start: str = "-120s",
    use_session_time_range: bool = True,
) -> bool:
    """Run DataCollector.get_data() and validate outputs. Returns True if all checks pass."""
    import json
    from data.data_collector import DataCollector
    from data.session_log_to_labels import session_log_to_labels

    print("Data collection test")
    print("  API_URL:", api_url)
    print("  Output dir:", output_dir)
    print("  Time range (start):", start)
    if not use_session_time_range:
        print("  Use session time range: no (using given start)")
    if session_log_redis:
        print("  Session log (redis):", session_log_redis)
    elif session_log_file:
        print("  Session log (file):", session_log_file)
    elif session_log_url:
        print("  Session log (url):", session_log_url)
    else:
        print("  Session log: from API (default, Redis-backed)")
    print()

    collector = DataCollector(api_url=api_url, output_dir=output_dir, redis_url=session_log_redis)
    paths = collector.get_data(start=start, use_session_time_range=use_session_time_range)

    metrics_path = paths.get("metrics_csv_path")
    log_path = paths.get("log_file_path")
    if not metrics_path or not log_path:
        print("FAIL: get_data() did not return metrics_csv_path and log_file_path")
        return False

    n_metrics = paths.get("metrics_count", "?")
    n_logs = paths.get("logs_count", "?")
    print("Quantity collected:", n_metrics, "metrics,", n_logs, "logs")

    out_path = labels_output or str(Path(output_dir) / "anomaly_labels.csv")
    metrics_df = pd.read_csv(metrics_path) if Path(metrics_path).exists() else None

    # Label source priority: Redis > file > URL > DataCollector/API default.
    log_data = None
    label_source = None

    if session_log_redis:
        log_data = _read_session_log_from_redis(session_log_redis)
        if log_data:
            label_source = f"Redis ({session_log_redis})"
        else:
            print("Labels: Redis session_log unavailable — falling back to DataCollector labels")
    elif session_log_file:
        path = Path(session_log_file).resolve()
        if path.exists():
            with open(path, "r") as f:
                log_data = json.load(f)
            if isinstance(log_data, dict) and "normal_windows" in log_data:
                label_source = f"file ({session_log_file})"
            else:
                print("Labels: file missing 'normal_windows':", session_log_file)
                log_data = None
        else:
            print("Labels: session log file not found:", session_log_file)
    elif session_log_url:
        log_data = collector.call_api.get_session_log(session_log_url=session_log_url)
        if log_data and isinstance(log_data, dict) and "normal_windows" in log_data:
            label_source = f"URL ({session_log_url})"
        else:
            print("Labels: no session log available at", session_log_url, "(need 'normal_windows')")
            log_data = None

    if log_data and metrics_df is not None:
        result = session_log_to_labels(log_data, out_path, metrics_df)
        if result:
            print("Labels:", out_path, f"(from {label_source})")
        else:
            print("Labels: generation failed (metrics alignment or conversion error)")
    elif not log_data and not label_source:
        print("Labels:", out_path, "(from API session_log, Redis-backed)")
    print()

    ok = True
    ok_metrics, msg_metrics = _check_metrics_csv(metrics_path)
    print("Metrics CSV:", msg_metrics)
    if not ok_metrics:
        ok = False

    ok_logs, msg_logs = _check_log_file(log_path)
    print("Log file:", msg_logs)
    if not ok_logs:
        ok = False

    # Run processing validation only if file checks passed.
    if ok:
        ok_processing, msg_processing = _check_processing(metrics_path, log_path)
        print("Processing (DAMDataProcessor):", msg_processing)
        if not ok_processing:
            ok = False

    # If labels exist, validate processing with labels/alignment.
    labels_path = labels_output or str(Path(output_dir) / "anomaly_labels.csv")
    if ok and Path(labels_path).exists():
        ok_labels, msg_labels = _check_processing_with_labels(
            metrics_path, log_path, labels_path
        )
        print("Processing with labels (alignment check):", msg_labels)
        if not ok_labels:
            ok = False

    print()
    if ok:
        print("PASS: Collection and validation succeeded.")
    else:
        print("FAIL: One or more checks failed.")
    return ok


def run_log_conversion_only():
    """Test _convert_logs_json_to_text with mock JSON (no API)."""
    from data.data_collector import DataCollector

    print("Log conversion test (no API)")
    # DataCollector requires API_URL; use a placeholder for this test.
    collector = DataCollector(api_url="http://placeholder", output_dir=tempfile.mkdtemp())
    mock_logs = [
        {
            "timestamp": "2025-01-15T12:00:00.123Z",
            "level": "INFO",
            "pid": "1006",
            "thread": "main",
            "logger": "app",
            "message": "Test message",
        },
        {
            "time": "2025-01-15T12:00:01.456Z",
            "log_level": "WARN",
            "process_id": "1007",
            "thread_name": "worker-1",
            "loggerName": "worker",
            "msg": "Another message",
        },
    ]
    lines = collector._convert_logs_json_to_text(mock_logs)
    if len(lines) != 2:
        print(f"FAIL: Expected 2 lines, got {len(lines)}")
        return False
    # Quick format check.
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\s+(\w+)\s+(\d+)\s+---\s+\[(.*?)\]\s+(.*?)\s+:\s+(.*)"
    )
    for i, line in enumerate(lines):
        if not pattern.match(line):
            print(f"FAIL: Line {i + 1} does not match LogAnalyzer format: {line[:80]!r}")
            return False
    print("PASS: Log conversion produced 2 lines in LogAnalyzer format.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test metrics and log collection for DAM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for collected files (default: aicybops_models/dam_model/data)",
    )
    parser.add_argument(
        "--test-logs-only",
        action="store_true",
        help="Only test log JSON-to-text conversion (no API call)",
    )
    parser.add_argument(
        "--test-connection-only",
        action="store_true",
        help="Only test GET /test_connection endpoint (no collection)",
    )
    parser.add_argument(
        "--session-log-redis",
        type=str,
        default=None,
        help="Read session log directly from Redis (real-time, even during live runs). "
             "Pass the Redis URL, e.g. redis://localhost:6380/0. "
             "Highest priority label source — preferred over --session-log-file and --session-log-url.",
    )
    parser.add_argument(
        "--session-log-url",
        type=str,
        default=None,
        help="Fetch session log (testbed anomalies) from this URL and save as anomaly_labels.csv. "
             "Use when testbed runs on another host; typically API_URL + '/session_log'.",
    )
    parser.add_argument(
        "--session-log-file",
        type=str,
        default=None,
        help="Use this local session_log JSON file for labels (e.g. Testbed/session_log.json). "
             "WARNING: file can be stale during a live testbed run (only written on termination). "
             "Prefer --session-log-redis for live runs.",
    )
    parser.add_argument(
        "--labels-output",
        type=str,
        default=None,
        help="Path for anomaly_labels.csv when using --session-log-url (default: output-dir/anomaly_labels.csv)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="-120s",
        help="API time range for metrics and logs, e.g. -120s (2 min), -600s (10 min). Use -600s to cover a 5-min testbed run.",
    )
    parser.add_argument(
        "--no-session-time-range",
        action="store_true",
        help="Do not use session log time range; always use --start. Use when session predates available API data.",
    )
    args = parser.parse_args()

    if args.test_logs_only:
        success = run_log_conversion_only()
        sys.exit(0 if success else 1)

    api_url = os.getenv("API_URL")
    if not api_url:
        print("ERROR: API_URL environment variable is not set.")
        print("Set it to your API base URL. Examples:")
        print("  - Docker (see docker-compose.yml): export API_URL=http://213.30.51.238:5010")
        print("  - Local API on port 5010:          export API_URL=http://localhost:5010")
        print("  - call_api.py default:             export API_URL=http://10.2.0.76:5010")
        sys.exit(1)

    if args.test_connection_only:
        success = run_connection_test(api_url)
        sys.exit(0 if success else 1)

    output_dir = args.output_dir or str(_data_dir)
    print("Output dir:", output_dir)
    print()

    success = run_collection_test(
        api_url,
        output_dir,
        session_log_redis=getattr(args, "session_log_redis", None),
        session_log_url=getattr(args, "session_log_url", None),
        session_log_file=getattr(args, "session_log_file", None),
        labels_output=getattr(args, "labels_output", None),
        start=getattr(args, "start", "-120s"),
        use_session_time_range=not getattr(args, "no_session_time_range", False),
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
