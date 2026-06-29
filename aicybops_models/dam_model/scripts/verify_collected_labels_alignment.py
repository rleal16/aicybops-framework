#!/usr/bin/env python3
"""
Run the full pipeline (collect -> process -> verify alignment) and report unlabellable periods.

Requires API_URL. Optionally --config (default: configs/dam_config.json).
Exit 0 on success, 1 on config/API error or alignment failure.
"""

import argparse
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_dam_model = _script_dir.parent
if str(_dam_model) not in sys.path:
    sys.path.insert(0, str(_dam_model))

from processing.data_analysis import DAMDataProcessor
from data.session_log_to_labels import report_unlabellable_periods


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pipeline (collect + process + verify alignment), report unlabellable periods."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_dam_model / "configs" / "dam_config.json",
        help="Path to DAM config JSON",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    api_url = os.environ.get("API_URL")
    if not api_url or not api_url.strip():
        print("ERROR: API_URL environment variable is not set.")
        print("  Set it to the collect_metrics_api base URL (e.g. export API_URL=http://localhost:5010)")
        return 1

    print("=" * 60)
    print("  Verify collected labels alignment (after processing)")
    print("=" * 60)
    print(f"  API_URL : {api_url}")
    print(f"  Config  : {config_path}")
    print()

    print("Step 1: Collecting metrics, logs, and labels from API...")
    print("Step 2: Processing with DAMDataProcessor...")
    processor = DAMDataProcessor(
        metrics_csv_path="",
        log_file_path="",
        config_path=str(config_path),
        window_size=10,
        stride=1,
        align_freq="1s",
        use_api=True,
    )

    try:
        processor._process_data()
    except ValueError as e:
        if "alignment" in str(e).lower() or "label-metrics" in str(e).lower():
            print()
            print("  FAILED: Label-metrics alignment verification failed during processing.")
            print(f"  {e}")
            return 1
        raise
    except Exception as e:
        print(f"  ERROR during processing: {e}")
        raise

    print()
    print("Step 3: Alignment verification passed.")

    labels_csv_path = getattr(processor, "labels_csv_path", None)
    if labels_csv_path:
        print()
        print("Step 4: Unlabellable periods (no overlap with session_log):")
        periods = report_unlabellable_periods(labels_csv_path)
        if periods:
            for start_ts, end_ts in periods:
                print(f"  {start_ts} — {end_ts}")
        else:
            print("  (none — all timestamps were labelled)")
    else:
        print()
        print("Step 4: No labels CSV path (session log may be unavailable).")

    print()
    print("  Done. Pipeline and verification OK.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
