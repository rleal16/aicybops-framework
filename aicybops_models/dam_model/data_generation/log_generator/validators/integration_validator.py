from typing import Any, Dict
from ..data_classes import ValidationResult, TemplateValidationRules
from .template_validator import TemplateValidator
from .context_validator import ContextValidator


class IntegrationValidator:
    """Validates template-context compatibility and integration only."""

    @classmethod
    def validate_placeholder_consistency(cls, templates: Dict[str, Any], contexts: Dict[str, Any]) -> ValidationResult:
        """
        Validate that all template placeholders have corresponding context data.
        
        Args:
            templates: Template dictionary
            contexts: Contexts data for validation
            
        Returns:
            ValidationResult: Result of the validation
        """
        result = ValidationResult()
        
        # Extract placeholders from templates 
        # Filter out special placeholders that are handled dynamically
        placeholders = TemplateValidator.extract_placeholders(templates, True)
        context_keys = ContextValidator.extract_keys(contexts)
        
        # If no placeholders found, validation passes automatically
        if not placeholders:
            return result
        
        
        missing_placeholders = placeholders - context_keys
        
        if missing_placeholders:
            result.add_error(f"Placeholders {missing_placeholders} have no corresponding data in contexts")
            return result
        
        return result
