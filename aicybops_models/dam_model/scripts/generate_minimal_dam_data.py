"""
Generate minimal metrics.csv, logs.txt, and anomaly_labels.csv for DAM so the
service can run train/predict without host-mounted data (e.g. in Docker).
Writes to data_generation/data/generated/ and data_generation/logs/ under
the dam_model directory. Safe to run from repo root or from dam_model.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def _dam_model_root() -> Path:
    """Resolve aicybops_models/dam_model root from this script's location."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def main() -> int:
    root = _dam_model_root()
    generated_dir = root / "data_generation" / "data" / "generated"
    logs_dir = root / "data_generation" / "logs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 2024-11-15 00:00:00 UTC — same range as logs below so metrics/logs align
    base_ts_ms = 1731628800000
    num_ticks = 30
    measurements = [
        ("container_cpu_usage_seconds_total", "counter", None),
        ("container_memory_working_set_bytes", "gauge", None),
        ("container_blkio_device_usage_total", "counter", "Read"),
        ("container_blkio_device_usage_total", "counter", "Write"),
        ("container_network_receive_bytes_total", "counter", None),
    ]

    metrics_path = generated_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["_time", "_measurement", "counter", "gauge", "operation"])
        for i in range(num_ticks):
            t = base_ts_ms + i * 1000
            for meas, val_col, op in measurements:
                row = [t, meas, "", "", ""]
                if val_col == "counter":
                    row[2] = str(100 + i)
                else:
                    row[3] = str(2_000_000 + i * 1000)
                if op:
                    row[4] = op
                w.writerow(row)

    # One label per unique timestamp (metrics.csv has multiple rows per timestamp in long format)
    labels_path = generated_dir / "anomaly_labels.csv"
    unique_ts = []
    seen = set()
    with open(metrics_path, "r", encoding="utf-8") as mf:
        reader = csv.DictReader(mf)
        for row in reader:
            t = row["_time"]
            if t not in seen:
                seen.add(t)
                unique_ts.append(t)
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["_time", "anomaly_label"])
        for i, t in enumerate(unique_ts):
            label = 1 if i >= len(unique_ts) - 10 else 0
            w.writerow([t, label])

    log_path = logs_dir / "logs.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(num_ticks):
            ts = f"2024-11-15T00:00:{i:02d}.000Z"
            f.write(
                f"{ts} INFO 1006 --- [io-1010-exec-10] c.example.Logger : request id=abc{i} duration=10ms\n"
            )

    print(f"Wrote {metrics_path}, {labels_path}, {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
