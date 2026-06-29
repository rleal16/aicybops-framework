import re
from string import Formatter
from typing import Any, Dict, Set
from ..data_classes import ValidationResult, TemplateValidationRules


class TemplateValidator:
    """
    Validates template structure and syntax only.
    Focuses exclusively on template validation following Single Responsibility Principle.
    """
    
    rules = TemplateValidationRules()
    
    @classmethod
    def extract_placeholders(cls, templates: Dict[str, Any], exclude_special_placeholders : bool = False) -> Set[str]:
        """
        Extract all placeholders from templates.
        
        Args:
            templates: Dictionary containing template data
            
        Returns:
            Set of all placeholder names found in templates
        """
        placeholders = set()
        placeholder_open = cls.rules.placeholder_delimiter_open
        placeholder_close = cls.rules.placeholder_delimiter_close
        # Fallback regex: not used, but kept for reference
        placeholder_pattern = re.compile(re.escape(placeholder_open) + r'(.+?)' + re.escape(placeholder_close))

        def recurse_templates(structure: Any):
            """Recursively traverse the template structure."""
            if isinstance(structure, dict):
                for value in structure.values():
                    recurse_templates(value)
            elif isinstance(structure, list):
                for item in structure:
                    recurse_templates(item)
            elif isinstance(structure, str):
                # Prefer the standard formatter parser to correctly handle braces
                try:
                    for _lit, field_name, _spec, _conv in Formatter().parse(structure):
                        if field_name:
                            placeholders.add(field_name)
                except ValueError:
                    # Malformed brace sequence (e.g., "{incomplete") — ignore for extraction.

                    pass
        recurse_templates(templates)
        
        if exclude_special_placeholders:
            placeholders -= cls.rules.special_placeholders
        
        return placeholders

    @classmethod
    def _validate_placeholders(cls, templates: Dict[str, Any], result: ValidationResult) -> bool:
        """
        Enforce that all placeholders are simple identifiers: ^[A-Za-z_][A-Za-z0-9_]*$
        Disallow format specs, attribute/index access. Adds errors to result for violations.

        Args:
            templates: Dictionary containing template data
            result: ValidationResult object

        Returns:
            True if placeholders are valid, False otherwise
        """
        simple_identifier = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

        def recurse(structure: Any, path: str) -> bool:
            if isinstance(structure, dict):
                for key, value in structure.items():
                    recurse(value, f"{path}.{key}" if path else str(key))
            elif isinstance(structure, list):
                for idx, item in enumerate(structure):
                    recurse(item, f"{path}[{idx}]")
            elif isinstance(structure, str):
                for _, field_name, format_spec, _ in Formatter().parse(structure):
                    
                    if not field_name:
                        # if a template does not contain a placeholder, simply continue
                        continue
                    # Reject format specs and non-simple names
                    if format_spec:
                        result.add_error(
                            f"Format specifiers are not allowed in placeholders: '{{{field_name}:{format_spec}}}' at {path}. Use simple identifiers."
                        )
                        return False
                    # if the placeholder is not a simple identifier, add an error
                    if not simple_identifier.match(field_name):
                        result.add_error(
                            f"Invalid placeholder '{{{field_name}}}' at {path}. Must start with a letter or underscore and contain only letters, numbers, and underscores. Use simple identifiers."
                        )
                        return False
                return True
            return False

        return recurse(templates, path="templates")
    @classmethod
    def _validate_subcategories(cls, subcategories: Dict[str, Any], result: ValidationResult) -> bool:
        """
        Validate template subcategory structure (dict of list of strings).
        Placeholders are optional - templates can be static.
        
        Args:
            subcategories: Dictionary containing subcategory data
            result: ValidationResult object
            
        Returns:
            True if structure is valid, False otherwise
        """
        if not isinstance(subcategories, dict):
            result.add_error("Subcategories must be a dictionary")
            return False

        if not subcategories:
            result.add_error("Subcategories dictionary cannot be empty")
            return False

        for subcategory_name, template_list in subcategories.items():
            if not isinstance(template_list, list):
                result.add_error(f"Subcategory '{subcategory_name}' must be a list")
                return False
            
            if not template_list:
                result.add_warning(f"Subcategory '{subcategory_name}' is empty")
                continue
            
            for i, template in enumerate(template_list):
                if not isinstance(template, str):
                    result.add_error(f"Template {i} in subcategory '{subcategory_name}' must be a string")
                    return False
                    
                if not template.strip():
                    result.add_error(f"Template {i} in subcategory '{subcategory_name}' must not be empty")
                    return False
                
        return True 

    @classmethod
    def validate_structure(cls, templates: Dict[str, Any]) -> ValidationResult:
        """
        Validate template structure independently.
        
        Args:
            templates: Template dictionary
            
        Returns:
            ValidationResult: Result of the validation
        """
        result = ValidationResult()
        
        # Basic structure validation
        if not isinstance(templates, dict):
            result.add_error("Templates must be a dictionary")
            return result

        if not templates:
            result.add_warning("Templates dictionary cannot be empty")
            return result

        required_template_categories = cls.rules.required_template_categories
        for category in templates.keys():
            if category not in required_template_categories:
                result.add_error(f"Invalid template category: '{category}'. Valid categories are: {required_template_categories}")
                return result

        for category, subcategories in templates.items():
            if not isinstance(subcategories, dict):
                result.add_error(f"Template category '{category}' must be a dictionary")
                continue
                
            if not cls._validate_subcategories(subcategories, result):
                return result

        # Strict placeholder validation across all template strings
        cls._validate_placeholders(templates, result)

        return result
