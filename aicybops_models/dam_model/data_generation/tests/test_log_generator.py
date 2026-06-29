import pytest
import json
import tempfile
import datetime
from pathlib import Path
from unittest.mock import patch, mock_open
from aicybops_models.dam_model.data_generation.log_generator import LogGenerator


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def valid_root_config(temp_dir):
    """Create a valid root configuration structure for LogGenerator."""
    # Create directories
    (temp_dir / "templates").mkdir()
    (temp_dir / "rules").mkdir()
    
    # Create contexts file with all required categories
    contexts = {
        "common_log_fields": {
            "level": ["INFO", "WARN", "ERROR"],
            "pid": [1001, 1002],
            "thread": ["thread-1"],
            "logger": ["com.example.service"]
        },
        "http_operations": {
            "method": ["GET", "POST"],
            "endpoint": ["/api/users"],
            "client_ip": ["192.168.1.1"],
            "status_code": [200, 500]
        },
        "database_operations": {
            "query_type": ["SELECT"],
            "table_name": ["users"]
        },
        "performance_monitoring": {
            "metric": ["cpu_usage"],
            "value": [85]
        },
        "business_logic": {
            "user_id": ["user_001"],
            "order_id": ["ord_001"]
        },
        "error_handling": {
            "error_type": ["ValidationError"],
            "error_message": ["Invalid input"]
        },
        "security_events": {
            "auth_result": ["successful"],
            "violation_type": ["SQL injection attempt"]
        }
    }
    (temp_dir / "contexts.json").write_text(json.dumps(contexts))
    
    # Create event_driven templates
    event_templates = {
        "http_operations": {
            "request_received": ["{timestamp} {level} {pid} --- [{thread}] {logger} : {method} {endpoint} from {client_ip}"],
            "response_sent": ["{timestamp} {level} {pid} --- [{thread}] {logger} : {method} {endpoint} - Response: {status_code}"]
        },
        "database_operations": {
            "query_execution": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Executing {query_type} query: {table_name}"]
        },
        "business_logic": {
            "order_processing": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Order created: {order_id} - User: {user_id}"]
        }
    }
    (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
    
    # Create state_driven templates
    state_templates = {
        "performance_monitoring": {
            "resource_alerts": ["{timestamp} {level} {pid} --- [{thread}] {logger} : High {metric} usage detected: {value}%"]
        },
        "error_handling": {
            "application_errors": ["{timestamp} {level} {pid} --- [{thread}] {logger} : Error processing request: {error_type}"]
        },
        "security_events": {
            "authentication": ["{timestamp} {level} {pid} --- [{thread}] {logger} : User authentication: {user_id} - {auth_result}"]
        }
    }
    (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
    
    # Create event_driven rules
    event_rules = {
        "network_tx": {
            "category": "http_operations",
            "subcategory": "response_sent",
            "rate_factor": 0.01,
            "min_logs": 0,
            "max_logs": 10
        }
    }
    (temp_dir / "rules" / "event_driven.json").write_text(json.dumps(event_rules))
    
    # Create state_driven rules
    state_rules = {
        "cpu_usage": {
            "warn": {
                "category": "performance_monitoring",
                "subcategory": "resource_alerts",
                "level": "WARN",
                "threshold": 80.0,
                "persistence_steps": 3,
                "cooldown_steps": 5
            }
        }
    }
    (temp_dir / "rules" / "state_driven.json").write_text(json.dumps(state_rules))
    
    # Create root config with correct structure
    root_config = {
        "templates": {
            "event_driven": ["templates/event_driven.json"],
            "state_driven": ["templates/state_driven.json"]
        },
        "rules": {
            "event_driven": ["rules/event_driven.json"],
            "state_driven": ["rules/state_driven.json"]
        },
        "contexts": ["contexts.json"]
    }
    root_path = temp_dir / "root_config.json"
    root_path.write_text(json.dumps(root_config))
    
    return root_path


@pytest.fixture
def log_generator_with_config(valid_root_config):
    """Create LogGenerator instance with valid root config."""
    return LogGenerator(valid_root_config)


# ============================================================================
# Test Classes - Organized by Functionality
# ============================================================================

class TestLogGeneratorInitialization:
    """Tests for LogGenerator initialization and configuration loading."""

    def test_init_with_valid_root_config(self, valid_root_config):
        """Test LogGenerator initialization with valid root config."""
        
        log_gen = LogGenerator(valid_root_config)
        
        assert log_gen.config_loader is not None
        assert log_gen.templates is not None
        assert log_gen.rules is not None
        assert log_gen.contexts is not None
        assert isinstance(log_gen.state_history, dict)
        assert isinstance(log_gen.correlation_ids, dict)

    def test_init_with_none_raises_error(self):
        """Test that passing None raises ValueError."""
        with pytest.raises(ValueError, match="root_config_path is required"):
            LogGenerator(None)

    def test_init_with_nonexistent_file_raises_error(self, temp_dir):
        """Test that nonexistent file raises FileNotFoundError."""
        nonexistent = temp_dir / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            LogGenerator(nonexistent)

    def test_init_with_invalid_config_raises_error(self, temp_dir):
        """Test that invalid config raises ValueError."""
        invalid_config = temp_dir / "invalid.json"
        invalid_config.write_text('{"invalid": "structure"}')
        with pytest.raises(ValueError, match="Configuration validation failed"):
            LogGenerator(invalid_config)

    def test_rules_structure_validation(self, log_generator_with_config):
        """Test that loaded rules have expected structure."""
        rules = log_generator_with_config.rules
        
        # Check event-driven rules
        assert "event_driven" in rules
        assert "network_tx" in rules["event_driven"]
        network_rule = rules["event_driven"]["network_tx"]
        assert network_rule["category"] == "http_operations"
        assert network_rule["subcategory"] == "response_sent"
        assert network_rule["rate_factor"] == 0.01
        
        # Check state-driven rules
        assert "state_driven" in rules
        assert "cpu_usage" in rules["state_driven"]
        cpu_rule = rules["state_driven"]["cpu_usage"]["warn"]
        assert cpu_rule["threshold"] == 80.0
        assert cpu_rule["persistence_steps"] == 3


class TestLogGeneratorStateTracking:
    """Tests for state tracking, statistics, and internal state management."""

    def test_state_tracking_initialization(self, log_generator_with_config):
        """Test that state tracking is properly initialized."""
        # Assert
        assert isinstance(log_generator_with_config.state_history, dict)
        assert isinstance(log_generator_with_config.correlation_ids, dict)
        assert isinstance(log_generator_with_config.generation_stats, dict)
        
        # Check generation stats structure
        assert 'total_logs' in log_generator_with_config.generation_stats
        assert 'templates_used' in log_generator_with_config.generation_stats
        assert 'correlation_strength' in log_generator_with_config.generation_stats
        assert 'template_diversity' in log_generator_with_config.generation_stats
        
        # Check initial values
        assert log_generator_with_config.generation_stats['total_logs'] == 0
        assert isinstance(log_generator_with_config.generation_stats['templates_used'], set)
        assert log_generator_with_config.generation_stats['correlation_strength'] == 0.0
        assert log_generator_with_config.generation_stats['template_diversity'] == 0.0

    def test_generation_stats_initial_state(self, log_generator_with_config):
        """Test that generation stats are in correct initial state."""
        stats = log_generator_with_config.generation_stats
        
        assert stats['total_logs'] == 0
        assert len(stats['templates_used']) == 0
        assert stats['correlation_strength'] == 0.0
        assert stats['template_diversity'] == 0.0

    def test_correlation_ids_initial_state(self, log_generator_with_config):
        """Test that correlation_ids is empty initially."""
        assert len(log_generator_with_config.correlation_ids) == 0

    def test_state_history_initial_state(self, log_generator_with_config):
        """Test that state_history is empty initially."""
        assert len(log_generator_with_config.state_history) == 0


class TestLogGeneratorTemplateSelection:
    """Tests for template selection functionality."""

    def test_select_template_success_event_driven(self, log_generator_with_config):
        """Test template selection from event_driven templates."""
        template = log_generator_with_config._select_template("event_driven", "http_operations", "request_received")
        assert template is not None
        assert "{timestamp}" in template
        assert "{method}" in template
        assert "{endpoint}" in template

    def test_select_template_success_state_driven(self, log_generator_with_config):
        """Test template selection from state_driven templates."""
        template = log_generator_with_config._select_template("state_driven", "performance_monitoring", "resource_alerts")
        assert template is not None
        assert "{timestamp}" in template
        assert "{metric}" in template
        assert "{value}" in template

    def test_select_template_invalid_template_type_raises_error(self, log_generator_with_config):
        """Test that invalid template_type raises KeyError."""
        with pytest.raises(KeyError, match="Template type 'invalid_type' not found"):
            log_generator_with_config._select_template("invalid_type", "http_operations", "request_received")

    def test_select_template_invalid_category_raises_error(self, log_generator_with_config):
        """Test that invalid category raises KeyError."""
        with pytest.raises(KeyError, match="Template category 'invalid' not found in template type 'event_driven'"):
            log_generator_with_config._select_template("event_driven", "invalid", "subcategory")

    def test_select_template_invalid_subcategory_raises_error(self, log_generator_with_config):
        """Test that invalid subcategory raises KeyError."""
        with pytest.raises(KeyError, match="Template subcategory 'invalid' not found"):
            log_generator_with_config._select_template("event_driven", "http_operations", "invalid")

    def test_select_template_wrong_template_type_for_category(self, log_generator_with_config):
        """Test that using wrong template_type for a category raises KeyError."""
        # http_operations is in event_driven, not state_driven
        with pytest.raises(KeyError, match="Template category 'http_operations' not found in template type 'state_driven'"):
            log_generator_with_config._select_template("state_driven", "http_operations", "request_received")


class TestLogGeneratorContext:
    """Tests for context value generation and retrieval."""

    def test_get_context_value_from_category(self, log_generator_with_config):
        """Test getting context value from specific category."""
        value = log_generator_with_config._get_context_value("method", "http_operations")
        assert value in ["GET", "POST"]

    def test_get_context_value_from_common_fields(self, log_generator_with_config):
        """Test getting context value from common_log_fields."""
        value = log_generator_with_config._get_context_value("level", "http_operations")
        assert value in ["INFO", "WARN", "ERROR"]

    def test_get_context_value_not_found_raises_error(self, log_generator_with_config):
        """Test that missing placeholder raises KeyError."""
        with pytest.raises(KeyError, match="Placeholder 'nonexistent' not found"):
            log_generator_with_config._get_context_value("nonexistent", "http_operations")


class TestLogGeneratorTimestamp:
    """Tests for timestamp extraction and distribution."""

    def test_extract_timestamp_missing_key_raises_error(self, log_generator_with_config):
        """Test that missing timestamp key raises KeyError."""
        metric_row = {'service_name': 'test-service', 'cpu_usage': 85.0}
        
        with pytest.raises(KeyError, match="Timestamp is required"):
            log_generator_with_config._extract_timestamp(metric_row)

    def test_extract_timestamp_none_value_raises_error(self, log_generator_with_config):
        """Test that None timestamp value raises KeyError."""
        metric_row = {'timestamp': None, 'service_name': 'test-service'}
        
        with pytest.raises(KeyError, match="Timestamp cannot be None"):
            log_generator_with_config._extract_timestamp(metric_row)

    def test_extract_timestamp_valid_datetime(self, log_generator_with_config):
        """Test extraction with valid datetime object."""
        expected_time = datetime.datetime(2023, 12, 25, 14, 30, 45)
        metric_row = {'timestamp': expected_time, 'service_name': 'test-service'}
        
        result = log_generator_with_config._extract_timestamp(metric_row)
        assert result == expected_time

    def test_calculate_offset_ms_single_log(self, log_generator_with_config):
        """Test offset calculation for single log."""
        offset = log_generator_with_config._calculate_offset_ms(0, 1)
        assert offset == 0

    def test_calculate_offset_ms_two_logs(self, log_generator_with_config):
        """Test offset calculation for two logs."""
        offset_0 = log_generator_with_config._calculate_offset_ms(0, 2)
        offset_1 = log_generator_with_config._calculate_offset_ms(1, 2)
        assert offset_0 == 0
        assert offset_1 == 1000

    def test_calculate_offset_ms_five_logs(self, log_generator_with_config):
        """Test offset calculation for five logs."""
        offsets = [log_generator_with_config._calculate_offset_ms(i, 5) for i in range(5)]
        expected = [0, 250, 500, 750, 1000]
        assert offsets == expected

    def test_distribute_timestamps_empty_list(self, log_generator_with_config):
        """Test timestamp distribution with empty log list."""
        base_time = datetime.datetime(2023, 12, 25, 14, 30, 45)
        result = log_generator_with_config._distribute_timestamps([], base_time)
        assert result == []

    def test_distribute_timestamps_single_log(self, log_generator_with_config):
        """Test timestamp distribution with single log."""
        base_time = datetime.datetime(2023, 12, 25, 14, 30, 45)
        logs = ["{timestamp} INFO: Single log message"]
        
        result = log_generator_with_config._distribute_timestamps(logs, base_time)
        
        assert len(result) == 1
        assert "{timestamp}" not in result[0]  # Placeholder should be replaced
        assert "INFO: Single log message" in result[0]
        assert "2023-12-25T14:30:45" in result[0]  # Base time should be present

    def test_distribute_timestamps_multiple_logs(self, log_generator_with_config):
        """Test timestamp distribution with multiple logs."""
        base_time = datetime.datetime(2023, 12, 25, 14, 30, 45)
        logs = [
            "{timestamp} INFO: Log 1",
            "{timestamp} WARN: Log 2", 
            "{timestamp} ERROR: Log 3"
        ]
        
        result = log_generator_with_config._distribute_timestamps(logs, base_time)
        
        assert len(result) == 3
        # All placeholders should be replaced
        for log in result:
            assert "{timestamp}" not in log
        
        # All logs should contain the base time
        for log in result:
            assert "2023-12-25T14:30:45" in log
        
        # Logs should be distributed within the same second
        timestamps = []
        for log in result:
            # Extract timestamp part (first part before space)
            timestamp_str = log.split(' ')[0]
            timestamps.append(timestamp_str)
        
        # All timestamps should be different (due to random offset)
        assert len(set(timestamps)) == 3

    def test_distribute_timestamps_timestamp_format(self, log_generator_with_config):
        """Test that generated timestamps have correct ISO format."""
        base_time = datetime.datetime(2023, 12, 25, 14, 30, 45)
        logs = ["{timestamp} INFO: Test log"]
        
        result = log_generator_with_config._distribute_timestamps(logs, base_time)
        timestamp_str = result[0].split(' ')[0]
        
        # Should match ISO format: YYYY-MM-DDTHH:MM:SS.mmmZ
        import re
        iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z'
        assert re.match(iso_pattern, timestamp_str), f"Timestamp {timestamp_str} doesn't match ISO format"


class TestLogGeneratorTemplateValidation:
    """Tests for template validation and invalid template handling."""

    def test_invalid_template_with_format_spec_fails_validation(self, temp_dir):
        """Templates with format specs in placeholders should fail during validation."""
        (temp_dir / "templates").mkdir(exist_ok=True)
        (temp_dir / "rules").mkdir(exist_ok=True)

        # Minimal contexts
        contexts = {"common_log_fields": {"level": ["INFO"]}}
        (temp_dir / "contexts.json").write_text(json.dumps(contexts))

        # Event templates with invalid format spec - should fail validation
        event_templates = {"http_operations": {"request_received": ["{timestamp} {level} {pid:.2f}"]}}
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))

        # Empty state templates to satisfy structure
        state_templates = {"performance_monitoring": {"resource_alerts": ["{timestamp} {level}"]}}
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))

        # Minimal rules
        (temp_dir / "rules" / "event_driven.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "state_driven.json").write_text(json.dumps({}))

        root_config = {
            "templates": {"event_driven": ["templates/event_driven.json"], "state_driven": ["templates/state_driven.json"]},
            "rules": {"event_driven": ["rules/event_driven.json"], "state_driven": ["rules/state_driven.json"]},
            "contexts": ["contexts.json"],
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))

        with pytest.raises(ValueError, match=r"Format specifiers are not allowed"):
            LogGenerator(root_path)

    def test_invalid_template_with_attribute_access_fails_validation(self, temp_dir):
        (temp_dir / "templates").mkdir(exist_ok=True)
        (temp_dir / "rules").mkdir(exist_ok=True)

        contexts = {"common_log_fields": {"level": ["INFO"]}}
        (temp_dir / "contexts.json").write_text(json.dumps(contexts))

        event_templates = {"http_operations": {"request_received": ["{timestamp} {level} {user.name}"]}} # invalid attribute access - should fail validation
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
        state_templates = {"performance_monitoring": {"resource_alerts": ["{timestamp} {level}"]}}
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
        (temp_dir / "rules" / "event_driven.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "state_driven.json").write_text(json.dumps({}))
        root_config = {
            "templates": {"event_driven": ["templates/event_driven.json"], "state_driven": ["templates/state_driven.json"]},
            "rules": {"event_driven": ["rules/event_driven.json"], "state_driven": ["rules/state_driven.json"]},
            "contexts": ["contexts.json"],
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))

        with pytest.raises(ValueError, match=r"Invalid placeholder .* Use simple identifiers"):
            LogGenerator(root_path)

    def test_invalid_template_with_indexing_fails_validation(self, temp_dir):
        (temp_dir / "templates").mkdir(exist_ok=True)
        (temp_dir / "rules").mkdir(exist_ok=True)

        contexts = {"common_log_fields": {"level": ["INFO"]}}
        (temp_dir / "contexts.json").write_text(json.dumps(contexts))

        event_templates = {"http_operations": {"request_received": ["{timestamp} {level} {items[0]}"]}} # invalid indexing - should fail validation
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
        state_templates = {"performance_monitoring": {"resource_alerts": ["{timestamp} {level}"]}}
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
        (temp_dir / "rules" / "event_driven.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "state_driven.json").write_text(json.dumps({}))
        root_config = {
            "templates": {"event_driven": ["templates/event_driven.json"], "state_driven": ["templates/state_driven.json"]},
            "rules": {"event_driven": ["rules/event_driven.json"], "state_driven": ["rules/state_driven.json"]},
            "contexts": ["contexts.json"],
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))

        with pytest.raises(ValueError, match=r"Invalid placeholder .* Use simple identifiers"):
            LogGenerator(root_path)

    def test_invalid_template_with_bad_identifier_fails_validation(self, temp_dir):
        (temp_dir / "templates").mkdir(exist_ok=True)
        (temp_dir / "rules").mkdir(exist_ok=True)

        contexts = {"common_log_fields": {"level": ["INFO"]}}
        (temp_dir / "contexts.json").write_text(json.dumps(contexts))

        event_templates = {"http_operations": {"request_received": ["{timestamp} {level} {bad-name}"]}} # invalid identifier - should fail validation
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
        state_templates = {"performance_monitoring": {"resource_alerts": ["{timestamp} {level}"]}}
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
        (temp_dir / "rules" / "event_driven.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "state_driven.json").write_text(json.dumps({}))
        root_config = {
            "templates": {"event_driven": ["templates/event_driven.json"], "state_driven": ["templates/state_driven.json"]},
            "rules": {"event_driven": ["rules/event_driven.json"], "state_driven": ["rules/state_driven.json"]},
            "contexts": ["contexts.json"],
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))

        with pytest.raises(ValueError, match=r"Invalid placeholder .* Use simple identifiers"):
            LogGenerator(root_path)
