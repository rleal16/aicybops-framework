import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union


class ConfigFileLoader:
    """Handles loading and parsing of configuration files."""
    
    @staticmethod
    def resolve_paths(base_dir: Path, rel_paths: Union[List[str], str]) -> List[Path]:
        """Resolve relative paths against base directory."""
        
        return [base_dir / rel_path for rel_path in rel_paths]

    @staticmethod
    def load_contexts(context_paths: List[Path]) -> Dict[str, Any]:
        """Load and merge contexts JSON content."""
        merged_contexts = {}
        for context_path in context_paths:
            try:
                with open(context_path, 'r') as f:
                    context_data = json.load(f)
                    # Merge contexts - later files override earlier keys
                    merged_contexts.update(context_data)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Skip invalid files but continue processing, for now
                continue
        return merged_contexts

    @staticmethod
    def load_templates(templates_paths: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Load and merge templates JSON content by category."""
        merged_templates = {}
        for category, paths in templates_paths.items():
            merged_templates[category] = {}
            for template_path in paths:
                if not template_path.is_file():
                    logging.warning(f"File not found: {template_path}")
                    continue
                try:
                    with template_path.open('r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        if not isinstance(template_data, dict):
                            logging.warning(f"Invalid JSON structure in file: {template_path}")
                            continue
                        # Merge categories from template files
                        for template_category, subcategories in template_data.items():
                            if not isinstance(subcategories, dict):
                                logging.warning(f"Invalid subcategories structure in file: {template_path}, category: {template_category}")
                                continue
                            if template_category not in merged_templates[category]:
                                merged_templates[category][template_category] = {}
                            for subcategory, templates in subcategories.items():
                                if not isinstance(templates, list):
                                    logging.warning(f"Invalid templates list in file: {template_path}, category: {template_category}, subcategory: {subcategory}")
                                    continue
                                if subcategory not in merged_templates[category][template_category]:
                                    merged_templates[category][template_category][subcategory] = []
                                merged_templates[category][template_category][subcategory].extend(templates)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error in file {template_path}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing file {template_path}: {e}")
        return merged_templates

    @staticmethod
    def load_rules(rules_paths: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Load rules JSON content by category."""
        merged_rules = {}
        for category, paths in rules_paths.items():
            merged_rules[category] = {}
            for rule_path in paths:
                try:
                    with open(rule_path, 'r') as f:
                        rule_data = json.load(f)
                        # Merge rules by category
                        merged_rules[category].update(rule_data)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    # Skip invalid files but continue processing
                    continue
        return merged_rules
