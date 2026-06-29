from typing import Dict, Any
from ..data_classes import ValidationResult, LogLevel, StateDrivenValidationRules


class StateDrivenValidator:
    """Validates state-driven rules structure and content."""
    rules = StateDrivenValidationRules()
    
    @classmethod
    def validate(cls, rules: Dict[str, Any]) -> ValidationResult:
        """Validate state-driven rules structure and content."""
        result = ValidationResult()
        
        if not isinstance(rules, dict):
            result.add_error("State-driven rules must be a dictionary")
            return result

        for metric_name, metric_rules in rules.items():
            if not isinstance(metric_rules, dict):
                result.add_error(f"Rules for metric '{metric_name}' must be a dictionary")
                return result

            for state_name, state_rule in metric_rules.items():
                if not isinstance(state_rule, dict):
                    result.add_error(f"State rule '{state_name}' for metric '{metric_name}' must be a dictionary")
                    return result

                cls._validate_required_fields(metric_name, state_name, state_rule, result)

        return result

    @classmethod
    def _validate_level(cls, level: str, result: ValidationResult) -> bool:
        """Validate level."""
        try:
            LogLevel(level)
        except ValueError:
            result.add_error(f"Invalid log level '{level}'")
            return False
        return True


    @classmethod
    def _validate_required_fields(cls, metric_name: str, state_name: str, state_rule: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate required fields for a state rule."""

        for field in cls.rules.required_fields:
            if field not in state_rule:
                result.add_error(f"Missing required field '{field}' in state '{state_name}' for metric '{metric_name}'")
                return False

        for field, accepted_type in cls.rules.accepted_types_for_fields.items():
            if field in state_rule and not isinstance(state_rule[field], accepted_type):
                result.add_error(f"Field '{field}' for state '{state_name}' of metric '{metric_name}' must be {accepted_type}")
                return False

        for field, accepted_range in cls.rules.accepted_values_for_fields.items():
            if field == "level":
                if not cls._validate_level(state_rule[field], result):
                    return False
                continue
            if field in state_rule and accepted_range is not None:
                value = state_rule[field]
                min_val, max_val = accepted_range
                # Special handling for threshold: must be > 0
                # TODO: reflect this in the accepted_values_for_fields in StateDrivenValidationRules
                if field == "threshold":
                    if not (value > 0):
                        result.add_error(f"Threshold for state '{state_name}' of metric '{metric_name}' must be > 0")
                        return False
                else:
                    if not (min_val <= value <= max_val):
                        if field == "persistence_steps":
                            result.add_error(f"Persistence steps for state '{state_name}' of metric '{metric_name}' must be a positive integer")
                        elif field == "cooldown_steps":
                            result.add_error(f"Cooldown steps for state '{state_name}' of metric '{metric_name}' must be a positive integer")
                        else:
                            result.add_error(f"Field '{field}' for state '{state_name}' of metric '{metric_name}' must be >= {min_val}")
                        return False

        for cross_field_rule in cls.rules.cross_field_rules:
            if not cross_field_rule["condition"](state_rule):
                
                field_values = []
                for field in cross_field_rule["fields"]:
                    field_values.append(f"{field}={state_rule.get(field, 'N/A')}")
                error_msg = f"{cross_field_rule['error_message']} for state '{state_name}' of metric '{metric_name}' ({', '.join(field_values)})"
                result.add_error(error_msg)
                return False
        
        return True
