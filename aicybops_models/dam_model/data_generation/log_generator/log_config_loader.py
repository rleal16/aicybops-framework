
import json
from pathlib import Path
from typing import Dict, Any
from .config_validator import ConfigValidator
from .helpers.config_file_loader import ConfigFileLoader


class LogConfigLoader:
    """
    Handles loading and validating all log configuration files.
    
    """

    def __init__(self, root_config_path: Path):
        """
        Initialize the log config loader with validation.
        
        Args:
            root_config_path: Path to root_config.json file (required)
        
        Raises:
            ValueError: If root_config_path is None or validation fails
            FileNotFoundError: If root_config_path doesn't exist
        """
        if root_config_path is None:
            raise ValueError("root_config_path is required")
        
        if not root_config_path.exists():
            raise FileNotFoundError(f"Root config not found: {root_config_path}")
        
        self.root_config_path = root_config_path
        self.base_dir = root_config_path.parent

        validation_result = ConfigValidator.validate_from_root_file(root_config_path)
        if not validation_result.is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(validation_result.errors)
            raise ValueError(error_msg)

        with open(root_config_path, 'r') as f:
            self.root_config = json.load(f)

        self.templates = self._load_templates()
        self.rules = self._load_rules()
        self.contexts = self._load_contexts()
    
    def _load_templates(self) -> Dict[str, Any]:
        """
        Load templates from configuration files referenced in root_config.
        
        Returns:
            Dictionary of templates organized by category and subcategory
            
        Example structure:
            {
                'event_driven': {
                    'http_operations': {
                        'request_received': ['template1', 'template2', ...]
                    }
                },
                'state_driven': { ... }
            }
        """
        templates_paths = {}
        for category, paths in self.root_config['templates'].items():
            resolved_paths = ConfigFileLoader.resolve_paths(self.base_dir, paths)
            templates_paths[category] = resolved_paths
        
        return ConfigFileLoader.load_templates(templates_paths)
    
    def _load_rules(self) -> Dict[str, Any]:
        """
        Load rules from configuration files referenced in root_config.
        
        Returns:
            Dictionary of rules organized by category
            
        Example structure:
            {
                'event_driven': {
                    'network_tx': {
                        'category': 'http_operations',
                        'subcategory': 'response_sent',
                        'rate_factor': 0.01,
                        ...
                    }
                },
                'state_driven': { ... }
            }
        """
        rules_paths = {}
        for category, paths in self.root_config['rules'].items():
            resolved_paths = ConfigFileLoader.resolve_paths(self.base_dir, paths)
            rules_paths[category] = resolved_paths
        
        return ConfigFileLoader.load_rules(rules_paths)
    
    def _load_contexts(self) -> Dict[str, Any]:
        """
        Load context data from configuration files referenced in root_config.
        
        Returns:
            Dictionary of context data organized by category
            
        Example structure:
            {
                'common_log_fields': {
                    'level': ['DEBUG', 'INFO', ...],
                    'pid': [1001, 1002, ...]
                },
                'http_operations': { ... }
            }
        """
        contexts_paths = ConfigFileLoader.resolve_paths(
            self.base_dir,
            self.root_config['contexts']
        )
        
        return ConfigFileLoader.load_contexts(contexts_paths)
