
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

class LogLevel(Enum):
    """Enumeration of supported log levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


    def set_invalid(self):
        self.is_valid = False
    
    def set_valid(self):
        self.is_valid = True

    def add_error(self, error: str):
        self.errors.append(error)
        self.set_invalid()

    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def reset(self):
        self.errors.clear()
        self.warnings.clear()
        self.set_valid()

@dataclass
class StructureValidationRules:
    """Rules for validating the structure of the root config."""

    valid_extensions: List[str] = field(default_factory=lambda: [
        "json", "yaml", "yml"
    ])

    # Updated to match the new simplified structure in root_config.json
    required_root_config_categories: List[str] = field(default_factory=lambda: [
        "templates", "rules", "contexts"
    ])

    # These keys and values are based on the new simplified structure in root_config.json
    required_files: Dict[str, List[str]] = field(default_factory=lambda: {
        "templates": ["event_driven", "state_driven"],
        "rules": ["event_driven", "state_driven"],
        "contexts": []  # contexts is a list of file paths
    })

@dataclass
class TemplateValidationRules:
    """Rules for validating the structure of the templates."""
    # Special placeholders that are handled dynamically by the log generator
    SPECIAL_PLACEHOLDERS = {'timestamp', 'level', 'value', 'metric', 'service_name', 'threshold'}

    # Base mandatory category - always required
    _base_categories: List[str] = field(default_factory=lambda: ["common_log_fields"])
    
    # Additional categories that can be extended
    _additional_categories: List[str] = field(default_factory=lambda: [
        "http_operations",
        "database_operations", 
        "business_logic",
        "performance_monitoring",
        "error_handling",
        "security_events"
    ])
    
    # Dynamic categories that can be added at runtime
    _dynamic_categories: List[str] = field(default_factory=list)
    
    @property
    def special_placeholders(self) -> set:
        return set(self.SPECIAL_PLACEHOLDERS)


    @property
    def required_template_categories(self) -> List[str]:
        """
        Get all required template categories with common_log_fields always first.
        This ensures common_log_fields is mandatory and always present.
        """
        return self._base_categories + self._additional_categories + self._dynamic_categories
    
    def add_dynamic_category(self, category: str) -> None:
        """
        Add a dynamic category to the validation rules.
        This allows external sources to extend the categories.
        
        Args:
            category: Name of the category to add
        """
        if category not in self.required_template_categories:
            self._dynamic_categories.append(category)
    
    def add_dynamic_categories(self, categories: List[str]) -> None:
        """
        Add multiple dynamic categories to the validation rules.
        
        Args:
            categories: List of category names to add
        """
        for category in categories:
            self.add_dynamic_category(category)
    


    placeholder_delimiter_open: str = field(default_factory=lambda: "{")
    placeholder_delimiter_close: str = field(default_factory=lambda: "}")


@dataclass
class EventDrivenValidationRules:
    """Rules for validating the structure of the event-driven rules."""

    required_fields: List[str] = field(default_factory=lambda: [
        "category", "subcategory", "rate_factor", "min_logs", "max_logs"
    ])
    
    # Accepted types for fields
    accepted_types_for_fields: Dict[str, Any] = field(default_factory=lambda: {
        "category": str,
        "subcategory": str,
        "rate_factor": (int, float),
        "min_logs": int,
        "max_logs": int
    })
    # Accepted values for fields
    accepted_values_for_fields: Dict[str, Any] = field(default_factory=lambda: {
        "category": None,  # No specific value constraint, just type
        "subcategory": None,  # No specific value constraint, just type
        "rate_factor": (0, float("inf")),
        "min_logs": (0, float("inf")),
        "max_logs": (0, float("inf"))
    })

    cross_field_rules: List[Any] = field(default_factory=lambda: [
        {
            "name": "min_max_log_consistency",
            "fields": ["min_logs", "max_logs"],
            "condition": lambda rule: rule.get("min_logs", 0) <= rule.get("max_logs", 0),
            "error_message": "min_logs must be less than or equal to max_logs"
        }
    ])


@dataclass
class StateDrivenValidationRules:
    """Rules for validating the structure of state-driven rules."""
    
    required_fields: List[str] = field(default_factory=lambda: [
        "category", "subcategory", "level", "threshold", "persistence_steps", "cooldown_steps"
    ])
    
    accepted_types_for_fields: Dict[str, Any] = field(default_factory=lambda: {
        "category": str,
        "subcategory": str,
        "level": str,
        "threshold": (int, float),
        "persistence_steps": int,
        "cooldown_steps": int
    })
    
    accepted_values_for_fields: Dict[str, Any] = field(default_factory=lambda: {
        "category": None,
        "subcategory": None,
        "level": None,  # Validated separately via LogLevel enum
        "threshold": (0, float("inf")),
        "persistence_steps": (1, float("inf")),
        "cooldown_steps": (1, float("inf"))
    })
    
    cross_field_rules: List[Any] = field(default_factory=lambda: [
        {
            "name": "persistence_cooldown_relationship",
            "fields": ["persistence_steps", "cooldown_steps"],
            "condition": lambda rule: rule.get("persistence_steps", 1) <= rule.get("cooldown_steps", 1),
            "error_message": "persistence_steps must be less than or equal to cooldown_steps"
        },
        {
            "name": "threshold_persistence_relationship",
            "fields": ["threshold", "persistence_steps"],
            "condition": lambda rule: rule.get("threshold", 0) <= 100 or rule.get("persistence_steps", 1) >= 2,
            "error_message": "thresholds > 100 should have at least 2 persistence_steps"
        }
    ])


@dataclass
class ValidationRules:
    """Container for all validation rules."""

    structure_validation_rules: StructureValidationRules = field(default_factory=StructureValidationRules)
    template_validation_rules: TemplateValidationRules = field(default_factory=TemplateValidationRules)

    def get_all_valid_placeholders(self) -> List[str]:
        """Get all valid placeholders from the template validation rules."""
        return list(set(placeholder for sublist in self.template_validation_rules.placeholder_names.values() for placeholder in sublist))

    def get_placeholder_for_category(self, category: str) -> List[str]:
        """Get all valid placeholders for a given category."""
        return self.template_validation_rules.placeholder_names.get(category, [])