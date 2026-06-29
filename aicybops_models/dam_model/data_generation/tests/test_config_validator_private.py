import json
import pytest
from pathlib import Path
from aicybops_models.dam_model.data_generation.log_generator.config_validator import ConfigValidator
from aicybops_models.dam_model.data_generation.log_generator.validators.template_validator import TemplateValidator
from aicybops_models.dam_model.data_generation.log_generator.validators.context_validator import ContextValidator
from aicybops_models.dam_model.data_generation.log_generator.validators.root_config_validator import RootConfigValidator
from aicybops_models.dam_model.data_generation.log_generator.helpers.config_file_loader import ConfigFileLoader
from aicybops_models.dam_model.data_generation.log_generator.validators.integration_validator import IntegrationValidator
from aicybops_models.dam_model.data_generation.log_generator.data_classes import ValidationResult


class TestConfigValidatorPrivateMethods:
    """Test suite for private methods of ConfigValidator """
    
    def test_validate_extensions(self):
        """Test the validation of file extensions."""
        # test valid extensions
        category_files = {
            "event_driven": ["templates/event.json", "templates/event.yaml"],
            "state_driven": ["templates/state.yaml"]
        }

        result = ValidationResult()
        is_valid = RootConfigValidator._validate_extensions(category_files, "templates", result)
        assert is_valid

        # test invalid extensions
        category_files = {
            "event_driven": ["templates/event.txt", "templates/event.yaml"],
            "state_driven": ["templates/state.yaml"]
        }
        result = ValidationResult()
        is_valid = RootConfigValidator._validate_extensions(category_files, "templates", result)
        assert not is_valid
        assert "Invalid extension" in result.errors[0]

    def test_validate_required_files_in_category(self):
        """Test the validation of required files in a category."""
        
        # test valid required files
        category_files = {
            "event_driven": ["templates/event.json"],
            "state_driven": ["templates/state.json"],
        }
        result = ValidationResult()
        is_valid = RootConfigValidator._validate_required_files_in_category(category_files, "templates", result)
        assert is_valid 
        assert len(result.errors) == 0

        # test invalid required files
        category_files = {
            "event_driven": ["templates/event.json"],
            # missing state_driven
        }
        result = ValidationResult()
        is_valid = RootConfigValidator._validate_required_files_in_category(category_files, "templates", result)
        assert not is_valid
        assert "Missing required file in category 'templates': 'state_driven'" in result.errors[0]
    
    def test_resolve_paths(self):
        """Test the resolution of paths."""
        base_dir = Path("/test/base")
        rel_paths = ["templates/event.json", "templates/state.yaml"]

        resolved_paths = ConfigFileLoader.resolve_paths(base_dir, rel_paths)

        expected_paths = [
            Path("/test/base/templates/event.json"),
            Path("/test/base/templates/state.yaml")
        ]

        assert resolved_paths == expected_paths

    def test_load_contexts(self, tmp_path: Path):
        """Test context loading and merging."""
        # create context files
        context1 = tmp_path / "context1.json"
        context2 = tmp_path / "context2.json"

        context1.write_text(json.dumps({
            "common_log_fields": {"level": ["INFO"], "pid": [1234]},
            "http_operations": {"method": ["GET"]}
        }))
        
        context2.write_text(json.dumps({
            "common_log_fields": {"timestamp": ["2025-01-01T12:00:00Z"]},
            "database_operations": {"query_type": ["SELECT"]}
        }))

        contexts = ConfigFileLoader.load_contexts([context1, context2])

        # The current implementation overwrites entire categories, not individual keys
        # So context2 completely replaces common_log_fields from context1
        assert "timestamp" in contexts["common_log_fields"]  # From context2
        assert "level" not in contexts["common_log_fields"]   # Overwritten by context2
        assert "pid" not in contexts["common_log_fields"]    # Overwritten by context2
        
        # http_operations from context1 should be preserved (not in context2)
        assert "method" in contexts["http_operations"]
        
        # database_operations from context2 should be present
        assert "query_type" in contexts["database_operations"]


class TestTemplateValidatorPrivateMethods:
    """Tests for TemplateValidator private helpers."""

    def test_extract_placeholders_basic(self):
        templates = {
            "http_operations": {
                "request_received": ["{timestamp} {method} {endpoint} from {client_ip}"],
                "response_sent": ["{timestamp} {status_code} {duration}ms"]
            },
            "database_operations": {
                "query_execution": ["{timestamp} {query_type} {table_name}"]
            }
        }
        placeholders = TemplateValidator.extract_placeholders(templates)
        assert placeholders == {"timestamp", "method", "endpoint", "client_ip", "status_code", "duration", "query_type", "table_name"}

    def test_extract_placeholders_edge_cases(self):
        # empty
        assert TemplateValidator.extract_placeholders({}) == set()
        # static messages
        templates = {"http_operations": {"request_received": ["static message"]}}
        assert TemplateValidator.extract_placeholders(templates) == set()
        # malformed mixed with valid
        templates = {"http_operations": {"request_received": ["{ok}", "{incomplete", "orphan}"]}}
        assert TemplateValidator.extract_placeholders(templates) == {"ok"}

    def test_validate_subcategories_valid(self):
        subs = {"request_received": ["{timestamp} {method}"], "response_sent": ["{timestamp} {status_code}"]}
        res = ValidationResult()
        assert TemplateValidator._validate_subcategories(subs, res)
        assert res.is_valid

    def test_validate_subcategories_invalid(self):
        subs = {"request_received": "not_a_list"}
        res = ValidationResult()
        assert not TemplateValidator._validate_subcategories(subs, res)
        assert not res.is_valid

        subs = {"request_received": []}
        res = ValidationResult()
        assert TemplateValidator._validate_subcategories(subs, res)
        assert res.is_valid
        assert any("is empty" in w for w in res.warnings)


class TestContextValidatorPrivateMethods:
    """Tests for ContextValidator private helpers."""

    def test_extract_keys_basic(self):
        contexts = {
            "common_log_fields": {"timestamp": ["t"], "level": ["INFO"]},
            "http_operations": {"method": ["GET"], "endpoint": ["/api"]}
        }
        keys = ContextValidator.extract_keys(contexts)
        assert keys == {"timestamp", "level", "method", "endpoint"}

    def test_extract_keys_edge_cases(self):
        # empty
        assert ContextValidator.extract_keys({}) == set()
        # non-dict category ignored; nested dict stays as key name
        contexts = {
            "category1": "not_a_dict",
            "common_log_fields": {"normal_range": {"cpu": "0-80%"}, "pid": [1234]}
        }
        keys = ContextValidator.extract_keys(contexts)
        assert keys == {"normal_range", "pid"}


class TestIntegrationValidatorPrivateMethods:
    """Tests for IntegrationValidator private helpers."""

    def test_validate_placeholder_consistency_valid(self):
        templates = {"http_operations": {"request_received": ["{timestamp} {method} {endpoint}"]}}
        contexts = {"common_log_fields": {"timestamp": ["t"]}, "http_operations": {"method": ["GET"], "endpoint": ["/api"]}}
        res = IntegrationValidator.validate_placeholder_consistency(templates, contexts)
        assert res.is_valid

    def test_validate_placeholder_consistency_missing_placeholder(self):
        templates = {"http_operations": {"request_received": ["{timestamp} {missing}"]}}
        contexts = {"common_log_fields": {"timestamp": ["t"]}}
        res = IntegrationValidator.validate_placeholder_consistency(templates, contexts)
        assert not res.is_valid
        assert any("missing" in e for e in res.errors)

class TestConfigValidatorRules:
    """Rule-focused tests."""

    def test_load_templates_merge(self, tmp_path: Path):
        a = tmp_path / "t_a.json"
        b = tmp_path / "t_b.json"
        # Each file should contain the category structure with subcategories
        a.write_text(json.dumps({"http_operations": {"request_received": ["A"]}}))
        b.write_text(json.dumps({"http_operations": {"request_received": ["B"]}}))
        merged = ConfigFileLoader.load_templates({"event_driven": [a, b]})
        # The method merges by subcategory within categories, so we should get concatenated lists
        assert merged["event_driven"]["http_operations"]["request_received"] == ["A", "B"]

    def test_load_rules_merge(self, tmp_path: Path):
        a = tmp_path / "r_a.json"
        b = tmp_path / "r_b.json"
        a.write_text(json.dumps({"network_tx": {"category": "http_operations", "subcategory": "request_received", "rate_factor": 0.1, "min_logs": 0, "max_logs": 1}}))
        b.write_text(json.dumps({"disk_read": {"category": "database_operations", "subcategory": "query_execution", "rate_factor": 0.1, "min_logs": 0, "max_logs": 1}}))
        merged = ConfigFileLoader.load_rules({"event_driven": [a, b]})
        assert set(merged["event_driven"].keys()) == {"network_tx", "disk_read"}

    def test_event_rules_valid_basic(self):
        rules = {"network_tx": {"category": "http_operations", "subcategory": "request_received", "rate_factor": 0.01, "min_logs": 0, "max_logs": 5}}
        res = ConfigValidator.validate_event_driven_rules(rules)
        assert res.is_valid

    def test_event_rules_invalid_cases(self):
        # negative rate - check for 'rate_factor' in error message
        rules = {"network_tx": {"category": "http_operations", "subcategory": "request_received", "rate_factor": -1, "min_logs": 0, "max_logs": 1}}
        res = ConfigValidator.validate_event_driven_rules(rules)
        assert not res.is_valid and any("rate_factor" in e for e in res.errors)
        
        # min>max - check for the exact error message from EventDrivenValidationRules
        rules = {"network_tx": {"category": "http_operations", "subcategory": "request_received", "rate_factor": 0, "min_logs": 5, "max_logs": 1}}
        res = ConfigValidator.validate_event_driven_rules(rules)
        assert not res.is_valid and any("min_logs must be less than or equal to max_logs" in e for e in res.errors)
        
        # missing fields
        rules = {"network_tx": {"category": "http_operations"}}
        res = ConfigValidator.validate_event_driven_rules(rules)
        assert not res.is_valid and any("Missing required field" in e for e in res.errors)
        
        # nonexistent category - this test should be removed since validation doesn't check templates
        # rules = {"network_tx": {"category": "nope", "subcategory": "request_received", "rate_factor": 0, "min_logs": 0, "max_logs": 1}}
        # res = ConfigValidator.validate_event_driven_rules(rules)
        # assert not res.is_valid and any("not found in templates" in e for e in res.errors)

    def test_state_rules_valid_basic(self):
        rules = {"cpu_usage": {"warn": {"category": "performance_monitoring", "subcategory": "resource_alerts", "level": "WARNING", "threshold": 80, "persistence_steps": 1, "cooldown_steps": 1}}}
        res = ConfigValidator.validate_state_driven_rules(rules)
        assert res.is_valid

    def test_state_rules_invalid_cases(self):
        # invalid level
        rules = {"cpu_usage": {"warn": {"category": "performance_monitoring", "subcategory": "resource_alerts", "level": "NOTALEVEL", "threshold": 80, "persistence_steps": 1, "cooldown_steps": 1}}}
        res = ConfigValidator.validate_state_driven_rules(rules)
        assert not res.is_valid and any("Invalid log level" in e for e in res.errors)
        # non-positive threshold
        rules = {"cpu_usage": {"warn": {"category": "performance_monitoring", "subcategory": "resource_alerts", "level": "WARNING", "threshold": 0, "persistence_steps": 1, "cooldown_steps": 1}}}
        res = ConfigValidator.validate_state_driven_rules(rules)
        assert not res.is_valid and any("Threshold" in e for e in res.errors)
        # non-positive steps
        rules = {"cpu_usage": {"warn": {"category": "performance_monitoring", "subcategory": "resource_alerts", "level": "WARNING", "threshold": 1, "persistence_steps": 0, "cooldown_steps": 1}}}
        res = ConfigValidator.validate_state_driven_rules(rules)
        assert not res.is_valid and any("Persistence steps" in e for e in res.errors)
        # missing fields
        rules = {"cpu_usage": {"warn": {"category": "performance_monitoring"}}}
        res = ConfigValidator.validate_state_driven_rules(rules)
        assert not res.is_valid and any("Missing required field" in e for e in res.errors)

    
    @pytest.mark.skip(reason="Not implemented")
    def test_validate_rules_with_templates_integration(self, tmp_path: Path):
        base = tmp_path
        ev = base / "event.json"
        st = base / "state.json"
        ev.write_text(json.dumps({
            "network_tx": {"category": "http_operations", "subcategory": "request_received", "rate_factor": 0.01, "min_logs": 0, "max_logs": 1}
        }))
        st.write_text(json.dumps({
            "cpu_usage": {"warn": {"category": "performance_monitoring", "subcategory": "resource_alerts", "level": "WARNING", "threshold": 80, "persistence_steps": 1, "cooldown_steps": 1}}
        }))
        root = {"rules": {"event_driven": ["event.json"], "state_driven": ["state.json"]}}
        res = IntegrationValidator.validate_rules_with_templates(root, base)
        assert res.is_valid
