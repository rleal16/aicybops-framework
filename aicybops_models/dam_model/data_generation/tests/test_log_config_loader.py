import pytest
import json
import tempfile
from pathlib import Path
from aicybops_models.dam_model.data_generation.log_generator.log_config_loader import LogConfigLoader


class TestLogConfigLoader:
    """Test suite for LogConfigLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def valid_root_config(self, temp_dir):
        """Create a valid root configuration structure."""
        # Create directories
        (temp_dir / "templates").mkdir()
        (temp_dir / "rules").mkdir()
        
        # Create context file with all required categories
        contexts = {
            "common_log_fields": {
                "level": ["INFO", "WARN", "ERROR"],
                "pid": [1001, 1002],
                "thread": ["thread-1", "thread-2"],
                "logger": ["com.example.service"]
            },
            "http_operations": {
                "method": ["GET", "POST"],
                "endpoint": ["/api/users", "/api/orders"],
                "status_code": [200, 201, 400, 500]
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
                "request_received": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : {method} {endpoint}"
                ]
            },
            "database_operations": {
                "query_execution": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : Executing {query_type} query: {table_name}"
                ]
            },
            "business_logic": {
                "order_processing": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : Order created: {order_id} - User: {user_id}"
                ]
            }
        }
        (temp_dir / "templates" / "event_driven.json").write_text(json.dumps(event_templates))
        
        # Create state_driven templates
        state_templates = {
            "performance_monitoring": {
                "resource_alerts": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : High {metric} usage: {value}%"
                ]
            },
            "error_handling": {
                "application_errors": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : Error processing request: {error_type}"
                ]
            },
            "security_events": {
                "authentication": [
                    "{timestamp} {level} {pid} --- [{thread}] {logger} : User authentication: {user_id} - {auth_result}"
                ]
            }
        }
        (temp_dir / "templates" / "state_driven.json").write_text(json.dumps(state_templates))
        
        # Create event_driven rules
        event_rules = {
            "network_tx": {
                "category": "http_operations",
                "subcategory": "request_received",
                "rate_factor": 0.01,
                "min_logs": 0,
                "max_logs": 5
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
        
        # Create root config
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

    def test_init_with_valid_config(self, valid_root_config):
        """Test initialization with valid configuration."""
        # Act
        loader = LogConfigLoader(valid_root_config)
        
        # Assert
        assert loader.root_config_path == valid_root_config
        assert loader.base_dir == valid_root_config.parent
        assert loader.root_config is not None
        assert loader.templates is not None
        assert loader.rules is not None
        assert loader.contexts is not None

    def test_init_with_none_path(self):
        """Test initialization with None path raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="root_config_path is required"):
            LogConfigLoader(None)

    def test_init_with_nonexistent_file(self, temp_dir):
        """Test initialization with nonexistent file raises FileNotFoundError."""
        # Arrange
        nonexistent_path = temp_dir / "nonexistent.json"
        
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Root config not found"):
            LogConfigLoader(nonexistent_path)

    def test_init_with_invalid_json(self, temp_dir):
        """Test initialization with invalid JSON raises ValueError."""
        # Arrange
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ invalid json }")
        
        # Act & Assert
        with pytest.raises(ValueError):
            LogConfigLoader(invalid_json)

    def test_templates_loaded_correctly(self, valid_root_config):
        """Test that templates are loaded with correct structure."""
        # Act
        loader = LogConfigLoader(valid_root_config)
        
        # Assert
        assert 'event_driven' in loader.templates
        assert 'state_driven' in loader.templates
        assert 'http_operations' in loader.templates['event_driven']
        assert 'request_received' in loader.templates['event_driven']['http_operations']
        assert len(loader.templates['event_driven']['http_operations']['request_received']) > 0

    def test_rules_loaded_correctly(self, valid_root_config):
        """Test that rules are loaded with correct structure."""
        # Act
        loader = LogConfigLoader(valid_root_config)
        
        # Assert
        assert 'event_driven' in loader.rules
        assert 'state_driven' in loader.rules
        assert 'network_tx' in loader.rules['event_driven']
        assert loader.rules['event_driven']['network_tx']['rate_factor'] == 0.01
        assert 'cpu_usage' in loader.rules['state_driven']
        assert 'warn' in loader.rules['state_driven']['cpu_usage']

    def test_contexts_loaded_correctly(self, valid_root_config):
        """Test that contexts are loaded with correct structure."""
        # Act
        loader = LogConfigLoader(valid_root_config)
        
        # Assert
        assert 'common_log_fields' in loader.contexts
        assert 'http_operations' in loader.contexts
        assert 'level' in loader.contexts['common_log_fields']
        assert 'method' in loader.contexts['http_operations']
        assert isinstance(loader.contexts['common_log_fields']['level'], list)

    def test_validation_catches_missing_category(self, temp_dir):
        """Test that validation catches missing required categories."""
        # Arrange - Create incomplete root config (missing 'contexts')
        root_config = {
            "templates": {
                "event_driven": ["templates/event.json"],
                "state_driven": ["templates/state.json"]
            },
            "rules": {
                "event_driven": ["rules/event.json"],
                "state_driven": ["rules/state.json"]
            }
            # Missing 'contexts' category
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))
        
        # Act & Assert
        with pytest.raises(ValueError, match="Configuration validation failed"):
            LogConfigLoader(root_path)

    def test_multiple_context_files_merged(self, temp_dir):
        """Test that multiple context files are properly merged."""
        # Arrange
        (temp_dir / "templates").mkdir()
        (temp_dir / "rules").mkdir()
        
        # Create two context files with all required categories
        context1 = {
            "common_log_fields": {"level": ["INFO"], "pid": [1001], "thread": ["thread-1"], "logger": ["com.example.service"]},
            "http_operations": {"method": ["GET"], "endpoint": ["/api/users"], "status_code": [200]},
            "database_operations": {"query_type": ["SELECT"], "table_name": ["users"]},
            "performance_monitoring": {"metric": ["cpu_usage"], "value": [85]},
            "business_logic": {"user_id": ["user_001"], "order_id": ["ord_001"]},
            "error_handling": {"error_type": ["ValidationError"], "error_message": ["Invalid input"]},
            "security_events": {"auth_result": ["successful"], "violation_type": ["SQL injection attempt"]}
        }
        (temp_dir / "context1.json").write_text(json.dumps(context1))
        
        context2 = {"http_operations": {"method": ["POST"], "endpoint": ["/api/orders"]}}
        (temp_dir / "context2.json").write_text(json.dumps(context2))
        
        # Create minimal template and rule files
        (temp_dir / "templates" / "event.json").write_text(json.dumps({}))
        (temp_dir / "templates" / "state.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "event.json").write_text(json.dumps({}))
        (temp_dir / "rules" / "state.json").write_text(json.dumps({}))
        
        # Create root config referencing both contexts
        root_config = {
            "templates": {
                "event_driven": ["templates/event.json"],
                "state_driven": ["templates/state.json"]
            },
            "rules": {
                "event_driven": ["rules/event.json"],
                "state_driven": ["rules/state.json"]
            },
            "contexts": ["context1.json", "context2.json"]
        }
        root_path = temp_dir / "root_config.json"
        root_path.write_text(json.dumps(root_config))
        
        # Act
        loader = LogConfigLoader(root_path)
        
        # Assert
        assert 'common_log_fields' in loader.contexts
        assert 'http_operations' in loader.contexts

    def test_integration_with_existing_configs(self):
        """Test integration with actual project configuration files."""
        # Arrange
        root_config_path = Path(
            "aicybops_models/dam_model/data_generation/log_generator/log_configs/root_config.json"
        )
        
        if not root_config_path.exists():
            pytest.skip("Actual config files not available")
        
        # Act
        loader = LogConfigLoader(root_config_path)
        
        # Assert
        assert 'event_driven' in loader.templates
        assert 'state_driven' in loader.templates
        assert 'event_driven' in loader.rules
        assert 'state_driven' in loader.rules
        assert 'common_log_fields' in loader.contexts
        assert 'http_operations' in loader.contexts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
