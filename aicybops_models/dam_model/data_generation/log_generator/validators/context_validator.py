from typing import Any, Dict, Set
from ..data_classes import ValidationResult, TemplateValidationRules


class ContextValidator:
    """
    Validates context data structure and completeness only.
    Focuses exclusively on context validation following Single Responsibility Principle.
    """
    
    rules = TemplateValidationRules()
    
    @classmethod
    def extract_keys(cls, contexts: Dict[str, Any]) -> Set[str]:
        """
        Extract all available context keys from the contexts data.
        These are the nested keys that can be used as placeholders in templates.
        
        Args:
            contexts: Dictionary containing all context data
            
        Returns:
            Set of all available context keys
        """
        context_keys = set()
        
        # Extract nested keys from each category
        for category, category_data in contexts.items():
            if isinstance(category_data, dict):
                # Add the nested keys (like 'levels', 'pids', 'methods', etc.)
                context_keys.update(category_data.keys())
        
        return context_keys

    @classmethod
    def validate_structure(cls, contexts: Dict[str, Any]) -> ValidationResult:
        """
        Validate context structure independently.
        
        Args:
            contexts: Context dictionary (from json file)
            
        Returns:
            ValidationResult: Result of the validation
        """
        result = ValidationResult()
        
        if not isinstance(contexts, dict):
            result.add_error("Contexts must be a dictionary")
            return result

        if not contexts:
            result.add_warning("Contexts dictionary cannot be empty")
            return result

        required_categories = cls.rules.required_template_categories
        for category in required_categories:
            if category not in contexts:
                result.add_error(f"Missing required context category: {category}")
                return result

        for category, context in contexts.items():
            if not isinstance(context, dict):
                result.add_error(f"Context for category '{category}' must be a dictionary")
                return result

            if not context:
                result.add_warning(f"Context for category '{category}' cannot be empty")
                continue

            # Each context key should map to a list (e.g., "methods": [...])
            for key, value in context.items():
                if not isinstance(value, list) and not isinstance(value, dict):
                    result.add_error(f"Context key '{key}' in category '{category}' must be a list or a dictionary")
                    return result
                # If it's a dict (e.g., "normal_ranges"), check its values are strings
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if not isinstance(subval, str):
                            result.add_error(f"Context subkey '{subkey}' in '{key}' of category '{category}' must be a string")
                            return result

        return result
