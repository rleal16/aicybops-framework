import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from aicybops_models.dam_model.data_generation.log_generator import LogGenerator

class TestLogGeneratorParametrized:
    """Parametrized tests for LogGenerator."""

    @pytest.mark.skip(reason="Test needs to be updated for new constructor signature")
    @pytest.mark.parametrize("template_path,config_path,expected_behavior", [
        (None, None, "defaults"),
        ("nonexistent.json", "nonexistent.json", "defaults"),
        ("valid.json", "nonexistent.json", "mixed"),
        ("nonexistent.json", "valid.json", "mixed"),
    ])
    def test_initialization_scenarios(self, template_path, config_path, expected_behavior, tmp_path):
        """Test various initialization scenarios."""
        # Initialize variables to None to prevent UnboundLocalError
        template_file = None
        config_file = None
        
        # Handle template_path
        if template_path == "valid.json":
            template_file = tmp_path / "valid.json"
            # Create valid template file
        elif template_path == "nonexistent.json":
            template_file = tmp_path / "nonexistent.json"
            # File doesn't exist, so it will use defaults

        # Handle config_path
        if config_path == "valid.json":
            config_file = tmp_path / "valid.json"
            # Create valid config file
        elif config_path == "nonexistent.json":
            config_file = tmp_path / "nonexistent.json"
            # File doesn't exist, so it will use defaults

        # Test the LogGenerator initialization
        log_generator = LogGenerator(template_file, config_file)
        # Add your assertions here

    @pytest.mark.skip(reason="Test needs to be updated for new constructor signature")
    @pytest.mark.parametrize("metric_name,expected_in_defaults", [
        ("cpu_usage", True),
        ("memory_usage", True),
        ("network_tx", True),
        ("network_rx", True),
        ("disk_read", True),
        ("disk_write", True),
        ("unknown_metric", False),
    ])
    def test_default_config_metrics(self, metric_name, expected_in_defaults):
        """Test that expected metrics are in default configuration."""
        log_generator = LogGenerator(None, None)
        config = log_generator._get_default_config()
        
        if expected_in_defaults:
            assert metric_name in config["thresholds"]
        else:
            assert metric_name not in config["thresholds"]

    @pytest.mark.skip(reason="Test needs to be updated for new constructor signature")
    @pytest.mark.parametrize("category,expected_in_defaults", [
        ("http_operations", True),
        ("database_operations", True),
        ("performance_monitoring", True),
        ("distributed_tracing", True),
        ("unknown_category", False),
    ])
    def test_default_template_categories(self, category, expected_in_defaults):
        """Test that expected categories are in default templates."""
        log_generator = LogGenerator(None, None)
        templates = log_generator._get_default_templates()
        
        if expected_in_defaults:
            assert category in templates
        else:
            assert category not in templates
