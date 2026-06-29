# Data Generation

Simple guide for generating metrics and logs for DAM model training and evaluation.

## Quick Start

### Generate Metrics AND Logs Together

Train a generator model and generate both metrics and logs:

```bash
python data_generator.py train-and-generate \
  --config-path configs/config_main_test.json
```

**What you need:**
- Config file (`configs/config_main_test.json`) - contains all settings including:
  - `data_path` - path to seed data CSV file (training data)
  - `output_paths.model` - path to save/load the model
  - `log_generator_config_path` - path to log generator root config
  - All other generation settings

**Output:**
- Metrics: `data/generated/metrics.csv` (from config)
- Logs: `logs/logs.txt` (from config)
- Model: `models/generator_model.pth`

### Generate Logs from Existing Metrics

If you already have metrics and just need to generate logs:

```bash
python generate_logs_from_metrics.py \
  --metrics-csv data/generated/metrics.csv \
  --config configs/config_main_test.json
```

**What you need:**
- Metrics CSV file
- Config file (must contain `root_config_path` and `output_paths.generated_logs`)

**Output:**
- Logs: path specified in `output_paths.generated_logs` in config

## Config File

All settings come from the config file. Required fields:

- `num_entities` - Number of entities to generate
- `segment_size` - Size of each entity's time series
- `sequence_index` - Timestamp column name (e.g., `_time`)
- `timestamp_format` - Format of timestamps (e.g., `unix_ms`)
- `entity_columns` - List of entity identifier columns
- `output_paths.generated_metrics` - Where to save metrics CSV
- `output_paths.generated_logs` - Where to save logs
- `log_generator_config_path` - Path to log generator root config (for log generation)

See `configs/config_main_test.json` for a complete example.

## Notes

- All paths and settings must be in the config file (no hardcoded defaults)
- Scripts will fail with clear errors if required config fields are missing
- Use `root_config_prometheus.json` for Prometheus-style metrics (counter, gauge)
- Use `root_config.json` for custom metrics (network_tx, network_rx, etc.)
