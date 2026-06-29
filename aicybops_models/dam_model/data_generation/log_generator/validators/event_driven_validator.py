from typing import Dict, Any
from ..data_classes import ValidationResult, EventDrivenValidationRules


class EventDrivenValidator:
    """Validates event-driven rules structure and content."""
    rules = EventDrivenValidationRules()

    @classmethod
    def _get_cross_field_error_message(cls, rule_error_message: str, metric_name: str, field1: str, field2: str, actual_value1: Any, actual_value2: Any) -> str:
        """Get cross-field error message."""
        return f"{rule_error_message} for '{metric_name}': {field1}={actual_value1}, {field2}={actual_value2}"

    @classmethod
    def _validate_required_fields(cls, metric_name: str, rule: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate required fields."""

        # check if all required fields are present
        required_fields = cls.rules.required_fields
        for field in required_fields:
            if field not in rule:
                result.add_error(f"Missing required field '{field}' in rule for metric '{metric_name}'")
                return False
        
        # check if all fields are of the correct type
        for field, accepted_type in cls.rules.accepted_types_for_fields.items():
            # TODO: to rethink this. By this point, we should have all required fields present. Possibly still needed, though...
            if field in rule and not isinstance(rule[field], accepted_type):
                result.add_error(f"Field '{field}' for metric '{metric_name}' must be a {accepted_type}")
                return False
            
        # validate field values/ranges
        for field, accepted_range in cls.rules.accepted_values_for_fields.items():
            if field in rule and accepted_range is not None:
                value = rule[field]
                min_val, max_val = accepted_range
                if not (min_val <= value <= max_val):
                    result.add_error(f"Field '{field}' for metric '{metric_name}' must be in {accepted_range}")
                    return False
            
        # validate cross-field rules
        for cross_field_rule in cls.rules.cross_field_rules:
            if not cross_field_rule["condition"](rule):
                rule_error_message = cross_field_rule["error_message"]
                field1 = cross_field_rule["fields"][0]
                field2 = cross_field_rule["fields"][1]
                actual_value1 = rule[field1]
                actual_value2 = rule[field2]
                msg = cls._get_cross_field_error_message(rule_error_message, metric_name, field1, field2, actual_value1, actual_value2)
                result.add_error(msg)
                return False
        
        return True

   
    @classmethod
    def validate(cls, rules: Dict[str, Any]) -> ValidationResult:
        """Validate event-driven rules structure and content."""
        result = ValidationResult()
        
        if not isinstance(rules, dict):
            result.add_error("Event-driven rules must be a dictionary")
            return result

        for metric_name, rule in rules.items():
            if not isinstance(rule, dict):
                result.add_error(f"Rule for metric '{metric_name}' must be a dictionary")
                continue

            cls._validate_required_fields(metric_name, rule, result)

        return result
