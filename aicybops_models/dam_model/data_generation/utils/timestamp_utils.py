"""
Timestamp conversion utilities for data generation.
"""
import datetime
from typing import Any, Optional


def format_datetime_to_timestamp(dt: datetime.datetime, timestamp_format: str) -> int | str:
    """Converts a datetime object TO the specified timestamp format (for generation)."""
    if timestamp_format == "unix_ms":
        return int(dt.timestamp() * 1000)
    elif timestamp_format in ["unix", "unix_s"]:
        return int(dt.timestamp())
    else:
        return dt.strftime('%Y-%m-%d %H:%M:%S')


def convert_to_datetime_for_logs(ts_value: Any, timestamp_format: str) -> datetime.datetime:
    """Convert timestamp value to datetime object for log generator compatibility."""
    if isinstance(ts_value, datetime.datetime):
        return ts_value
    elif isinstance(ts_value, (int, float)):
        if timestamp_format == "unix_ms":
            return datetime.datetime.fromtimestamp(ts_value / 1000)
        elif timestamp_format in ["unix", "unix_s"]:
            return datetime.datetime.fromtimestamp(ts_value)
        else:
            return datetime.datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
    elif isinstance(ts_value, str):
        try:
            return datetime.datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
        except ValueError:
            # Try other formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    return datetime.datetime.strptime(ts_value, fmt)
                except ValueError:
                    continue
            # Fallback to current time if all parsing fails
            return datetime.datetime.now()
    else:
        return datetime.datetime.now()


def generate_timestamp_for_row(
    ix: int, 
    entity_num: int, 
    segment_size: int, 
    start_date: Optional[str],
    timestamp_format: str,
    sequence_index: str
) -> Any:
    """Generate timestamp for a row based on entity number and position."""
    if start_date:
        base_start = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    else:
        base_start = datetime.datetime.now()
    
    position_in_entity = ix % segment_size
    total_minutes_offset = entity_num * segment_size + position_in_entity
    row_time = base_start + datetime.timedelta(minutes=total_minutes_offset)
    
    return format_datetime_to_timestamp(row_time, timestamp_format)
