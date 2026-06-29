import datetime
import json
import re
import pytest
import tempfile
from pathlib import Path
from aicybops_models.dam_model.data_generation.log_generator import LogGenerator

class TestLogGeneratorTimestamp:
    """ Tests for timestamp extraction and distribution in LogGenerator. """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def log_generator(self, temp_dir):
        """Fixture to create a LogGenerator instance."""
        # Create a minimal valid configuration for timestamp tests
        (temp_dir / "templates").mkdir()
        (temp_dir / "rules").mkdir()
        
        # Create minimal contexts
        contexts = {
            "common_log_fields": {
                "level": ["INFO"],
                "pid": [1001],
                "thread": ["thread-1"],
                "logger": ["com.example.service"]
            },
            "http_operations": {
                "method": ["GET"],
                "endpoint": ["/api/test"]
            },
            "database_operations": {
                "query_type": ["SELECT"],
                "table_name": ["test"]
            },
            "performance_monitoring": {
                "metric": ["cpu_usage"],
                "value": [50]
            },
            "business_logic": {
                "user_id": ["user_001"],
                "order_id": ["ord_001"]
            },
            "error_handling": {
                "error_type": ["TestError"]
            },
            "security_events": {
                "auth_result": ["successful"]
            }
        }
        (temp_dir / "contexts.json").write_text(json.dumps(contexts))
        
        # Create minimal event_driven templates
        event_templates = {
            "http_operations": {
                "request_received": ["{timestamp} {level} {pid} --- [{thread}] {logger} : {method} {endpoint}"]
            },
            "database_operations": {
                "query_execution": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Executing {query_type} query: {table_name}"]
            },
            "business_logic": {
                "order_processing": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Order created: {order_id} - User: {user_id}"]
            }
        }
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
        
        # Create minimal state_driven templates
        state_templates = {
            "performance_monitoring": {
                "resource_alerts": ["{timestamp} {level} {pid} --- [{thread}] {logger} : High {metric} usage: {value}%"]
            },
            "error_handling": {
                "application_errors": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Error processing request: {error_type}"]
            },
            "security_events": {
                "authentication": ["{timestamp} {level} {pid} --- [{thread}] {logger} : User authentication: {user_id} - {auth_result}"]
            }
        }
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
        
        # Create minimal rules
        rules = {
            "event_driven": {},
            "state_driven": {}
        }
        (temp_dir / "rules" / "rules.json").write_text(json.dumps(rules))
        
        # Create root config
        root_config = {
            "templates": {
                "event_driven": ["templates/event_driven.json"],
                "state_driven": ["templates/state_driven.json"]
            },
            "rules": {
                "event_driven": ["rules/rules.json"],
                "state_driven": ["rules/rules.json"]
            },
            "contexts": ["contexts.json"]
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))
        
        return LogGenerator(root_path)

    @pytest.mark.parametrize("input_ts,expected_type", [
        (datetime.datetime(2025, 9, 9, 12, 0, 0), datetime.datetime),
        ("2025-09-09T12:00:00Z", datetime.datetime),
        ("2025-09-09 12:00:00", datetime.datetime),
        (1694260800, datetime.datetime),  # Unix timestamp
    ])
    def test_extract_timestamp_various_formats(self, log_generator, input_ts, expected_type):
        """Test _extract_timestamp with various timestamp formats."""
        metric_row = {'timestamp': input_ts}
        ts = log_generator._extract_timestamp(metric_row)
        assert isinstance(ts, expected_type)
        # Should be close to the input if input is not None
        if input_ts and isinstance(input_ts, datetime.datetime):
            assert ts == input_ts
    
    def test_extract_timestamp_invalid_string(self, log_generator):
        """Test _extract_timestamp with invalid string falls back to now."""
        metric_row = {'timestamp': 'not-a-timestamp'}
        ts = log_generator._extract_timestamp(metric_row)
        assert isinstance(ts, datetime.datetime)
        # Should be close to now
        assert abs((ts - datetime.datetime.now()).total_seconds()) < 2

    def test_distribute_timestamps_basic(self, log_generator):
        """Test _distribute_timestamps distributes timestamps correctly."""
        logs = [
            "{timestamp} log1",
            "{timestamp} log2",
            "{timestamp} log3"
        ]
        base_time = datetime.datetime(2025, 9, 9, 12, 0, 0)
        timestamped_logs = log_generator._distribute_timestamps(logs, base_time)
        assert len(timestamped_logs) == 3
        for log in timestamped_logs:
            assert "log" in log
            # Check timestamp format
            ts_str = log.split(" ")[0]
            assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", ts_str) is not None
        
    
    def test_distribute_timestamps_empty(self, log_generator):
        """Test _distribute_timestamps returns empty list for empty input."""
        result = log_generator._distribute_timestamps([], datetime.datetime.now())
        assert result == []

    def test_distribute_timestamps_single_log(self, log_generator):
        """Test _distribute_timestamps with a single log entry."""
        logs = ["{timestamp} onlylog"]
        base_time = datetime.datetime(2025, 9, 9, 12, 0, 0)
        result = log_generator._distribute_timestamps(logs, base_time)
        assert len(result) == 1
        assert "onlylog" in result[0]
        ts_str = result[0].split(" ")[0]
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", ts_str) is not None