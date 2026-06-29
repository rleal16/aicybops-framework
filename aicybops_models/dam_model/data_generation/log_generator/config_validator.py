import json
from pathlib import Path
from typing import Any, Dict

from .data_classes import LogLevel, ValidationResult, ValidationRules, TemplateValidationRules, StructureValidationRules
from .validators.event_driven_validator import EventDrivenValidator
from .validators.state_driven_validator import StateDrivenValidator
from .validators.template_validator import TemplateValidator
from .validators.context_validator import ContextValidator
from .validators.root_config_validator import RootConfigValidator
from .validators.integration_validator import IntegrationValidator
from .helpers.config_file_loader import ConfigFileLoader



class ConfigValidator:
    """
    Master orchestrator for all configuration validation.
    Delegates to specialized validators and loaders.
    """

    rules = StructureValidationRules()


    @classmethod
    def validate_root_config_structure(cls, root_config: Dict[str, Any]) -> ValidationResult:
        """Validate root configuration structure."""
        return RootConfigValidator.validate(root_config)


    @classmethod
    def validate_event_driven_rules(cls, rules: Dict[str, Any]) -> ValidationResult:
        """Validate event-driven rules using dedicated validator."""
        return EventDrivenValidator.validate(rules)

    @classmethod
    def validate_state_driven_rules(cls, rules: Dict[str, Any]) -> ValidationResult:
        """Validate state-driven rules using dedicated validator."""
        return StateDrivenValidator.validate(rules)

    @classmethod
    def validate_placeholder_consistency(cls, category_templates, contexts):
        return IntegrationValidator.validate_placeholder_consistency(category_templates, contexts)
    
    @staticmethod
    def _merge_validation_results(dst: ValidationResult, src: ValidationResult):
        """Merge validation results from src into dst."""
        dst.errors.extend(src.errors)
        dst.warnings.extend(src.warnings)
        if not src.is_valid:
            dst.set_invalid()

    @classmethod
    def validate_from_root_file(cls, root_config_path: Path) -> ValidationResult:
        """
        Validate all configs from a root config file with full path resolution.
        """
          
        result = ValidationResult()

        try:
            with open(root_config_path, 'r') as f:
                root_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            result.add_error(f"Failed to load root config: {e}")
            return result

        root_result = RootConfigValidator.validate(root_config)
        if not root_result.is_valid:
            return root_result

        cls._merge_validation_results(result, root_result)

        base_dir = root_config_path.parent

        contexts_paths = ConfigFileLoader.resolve_paths(base_dir, root_config.get('contexts', []))
        contexts = ConfigFileLoader.load_contexts(contexts_paths)
        
        templates_paths = {}
        if 'templates' in root_config:
            for category, paths in root_config['templates'].items():
                templates_paths[category] = ConfigFileLoader.resolve_paths(base_dir, paths)
        templates = ConfigFileLoader.load_templates(templates_paths)

        for category_type, category_templates in templates.items():
            template_result = TemplateValidator.validate_structure(category_templates)
            cls._merge_validation_results(result, template_result)
            if not template_result.is_valid:
                result.set_invalid()

        context_result = ContextValidator.validate_structure(contexts)
        cls._merge_validation_results(result, context_result)
        if not context_result.is_valid:
            result.set_invalid()

        if context_result.is_valid and result.is_valid:
            for _, category_templates in templates.items():
                integration_result = cls.validate_placeholder_consistency(category_templates, contexts)
                cls._merge_validation_results(result, integration_result)
                if not integration_result.is_valid:
                    result.set_invalid()
        return result