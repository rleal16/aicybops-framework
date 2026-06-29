#!/usr/bin/env python3
"""
Start the testbed in the background, wait 30 seconds, then run DAM data collection
(DataCollector / test_collection) as usual.

Usage
-----
    # From dam_model/data or dam_model. Requires API_URL (e.g. collect_metrics_api on 5010).
    export API_URL=http://localhost:5010
    python run_testbed_then_collect.py

    # Or as module from repo root / AICybOps
    API_URL=http://localhost:5010 python -m aicybops_models.dam_model.data.run_testbed_then_collect

    # Custom wait and output dir
    API_URL=http://localhost:5010 python run_testbed_then_collect.py --wait 45 --output-dir /path/to/output

    # Testbed options are passed through (e.g. duration, --skip-setup)
    API_URL=http://localhost:5010 python run_testbed_then_collect.py --duration 5

    # Labels are fetched from API GET /session_log and written to output-dir/anomaly_labels.csv.
    # Ensure the API has the session log (SESSION_LOG_PATH or POST from testbed).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # dam_model/data
DAM_MODEL_DIR = SCRIPT_DIR.parent             # dam_model
REPO_ROOT = DAM_MODEL_DIR.parent.parent.parent  # workspace root (dam_model -> aicybops_models -> AICybOps -> root)
TESTBED_DIR = REPO_ROOT / "Testbed"


def main():
    parser = argparse.ArgumentParser(
        description="Start testbed in background, wait, then run DAM data collection."
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=30,
        help="Seconds to wait after starting testbed before collecting (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for metrics.csv and logs.txt (default: dam_model/data)",
    )
    args, testbed_args = parser.parse_known_args()

    api_url = os.environ.get("API_URL")
    if not api_url:
        print("ERROR: API_URL environment variable is not set.", file=sys.stderr)
        print("Example: export API_URL=http://localhost:5010", file=sys.stderr)
        sys.exit(1)

    if not (TESTBED_DIR / "run_testbed.py").exists():
        print(f"ERROR: run_testbed.py not found at {TESTBED_DIR}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or str(SCRIPT_DIR)

    # 1. Start testbed in background (with --detach so it logs to file and returns)
    testbed_cmd = [sys.executable, str(TESTBED_DIR / "run_testbed.py"), "--detach"] + testbed_args
    print("Starting testbed in background...")
    print(f"  Command: {' '.join(testbed_cmd)}")
    proc = subprocess.Popen(
        testbed_cmd,
        cwd=str(TESTBED_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate(timeout=15)
    if proc.returncode != 0:
        print(out or "Testbed failed to start.", file=sys.stderr)
        sys.exit(proc.returncode)
    if out:
        print(out.strip())

    # 2. Wait
    print(f"Waiting {args.wait} seconds before collecting...")
    time.sleep(args.wait)

    # 3. Run data collector with session log labels (test_collection uses DataCollector)
    env = {**os.environ, "API_URL": api_url}
    session_log_url = f"{api_url.rstrip('/')}/session_log"
    labels_output = os.path.join(output_dir, "anomaly_labels.csv")
    collect_cmd = [
        sys.executable, "-m", "data.test_collection",
        "--output-dir", output_dir,
        "--session-log-url", session_log_url,
        "--labels-output", labels_output,
    ]
    print(f"Running data collection: API_URL={api_url} output={output_dir} labels={labels_output}")
    result = subprocess.run(
        collect_cmd,
        cwd=str(DAM_MODEL_DIR),
        env=env,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
