import os
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def collect_data_from_api(
    config_path: str,
    use_session_time_range: bool = True,
    start: Optional[str] = None,
    training_window_minutes: int = 0,
) -> Dict[str, str]:
    """
    Collect metrics/logs/labels via API and return local file paths.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Prefer writable env dir (useful when config mounts are read-only).
    env_collect_dir = os.environ.get('DAM_API_COLLECT_DIR')
    if env_collect_dir:
        output_dir = env_collect_dir
    else:
        config_dir = Path(config_path).parent
        if 'output_paths' in config and 'generated_metrics' in config['output_paths']:
            generated_metrics_path = Path(config['output_paths']['generated_metrics'])
            if not generated_metrics_path.is_absolute():
                generated_metrics_path = config_dir / generated_metrics_path
            output_dir = str(generated_metrics_path.parent.parent / 'collected')
        else:
            output_dir = str(config_dir / 'data' / 'collected')

    _dam_model = Path(__file__).resolve().parent.parent
    try:
        from aicybops_models.dam_model.data.data_collector import DataCollector
    except (ModuleNotFoundError, ImportError):
        # Ensure dam_model/data is imported instead of /app/data (Docker).
        _dam_root = str(_dam_model)
        if _dam_root in sys.path:
            sys.path.remove(_dam_root)
        sys.path.insert(0, _dam_root)
        from data.data_collector import DataCollector

    data_collector = DataCollector(api_url=os.getenv('API_URL'), output_dir=output_dir)
    kwargs = {"use_session_time_range": use_session_time_range}
    if start is not None:
        kwargs["start"] = start
    if training_window_minutes > 0:
        kwargs["training_window_minutes"] = training_window_minutes
    return data_collector.get_data(**kwargs)
