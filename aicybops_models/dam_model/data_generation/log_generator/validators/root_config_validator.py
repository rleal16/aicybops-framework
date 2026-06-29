from typing import Any, Dict, List, Union

from ..data_classes import ValidationResult, StructureValidationRules


class RootConfigValidator:
    """
    Validates root configuration file structure and format.
    """

    rules = StructureValidationRules()

    @classmethod
    def _validate_file_paths_and_extensions(cls, file_paths: List[str], category: str, result: ValidationResult) -> bool:
        valid_extensions = cls.rules.valid_extensions
        for file_path in file_paths:
            if not isinstance(file_path, str):
                result.add_error(f"File path in '{category}' list must be a string")
                return False
            ext = file_path.split(".")[-1]
            if ext not in valid_extensions:
                result.add_error(f"Invalid extension for '{file_path}' in category '{category}'")
                return False
        return True

    @classmethod
    def _validate_extensions(cls, category_files: Union[Dict[str, Any], List[str]], category: str, result: ValidationResult) -> bool:
        """Validate file extensions."""
        
        if isinstance(category_files, list):
            if not cls._validate_file_paths_and_extensions(category_files, category, result):
                return False
        else:
            for file_category, file_list in category_files.items():
                # file_list is expected to be a list of file paths
                if not isinstance(file_list, list):
                    result.add_error(f"File value for '{file_category}' in category '{category}' must be a list")
                    return False
                if not cls._validate_file_paths_and_extensions(file_list, category, result):
                    return False
        return True

    @classmethod
    def _validate_required_files_in_category(cls, category_files: Union[Dict[str, Any], List[str]], category: str, result: ValidationResult) -> bool:
        """
        Validate if the category (in the root config) has the required files.

        Args:
            category_files: Dictionary containing the files in the category
            category: Name of the category
            result: ValidationResult object
        Returns:
            True if the category has the required files, False otherwise
        """
        
        required_files = cls.rules.required_files[category]
        
        # Special handling for contexts - it's a list, not a dict (for now)
        # TODO Should be handled in data_classes.py
        if category == "contexts":
            if not isinstance(category_files, list):
                result.add_error(f"Category '{category}' must be a list of file paths")
                return False
            if not category_files:
                result.add_warning(f"Category '{category}' is empty - no context files specified")
        else:
            # For templates and rules categories
            for file in required_files:
                if file not in category_files:
                    result.add_error(f"Missing required file in category '{category}': '{file}'")
                    return False
                if not isinstance(category_files[file], list):
                    result.add_error(f"File '{file}' in category '{category}' must be a list of file paths")
                    return False
                if not category_files[file]:
                    result.add_warning(f"File list for '{file}' in category '{category}' is empty")
        
        if not cls._validate_extensions(category_files, category, result):
            return False
        return True

    @classmethod
    def validate(cls, root_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate the structure of the root configuration file.

        Args:
            root_config: Root configuration dictionary

        Returns:
            ValidationResult: Result of the validation
        """
        
        result = ValidationResult()
        required_root_config_categories = cls.rules.required_root_config_categories

        if not isinstance(root_config, dict):
            result.add_error("Root configuration must be a dictionary")
            return result

        for category in required_root_config_categories:
            if category not in root_config:
                result.add_error(f"Missing required category in root config: {category}")
                return result
            
            if category == "contexts":
                if not isinstance(root_config[category], list):
                    result.add_error(f"Category '{category}' must be a list of file paths")
                    return result
                if not root_config[category]:
                    result.add_warning(f"Category '{category}' is empty - no context files specified")
            else:
                # For templates and rules - they should be dictionaries
                if not isinstance(root_config[category], dict):
                    result.add_error(f"Category '{category}' must be a dictionary")
                    return result

            if not cls._validate_required_files_in_category(
                root_config[category], category, result
            ):
                return result

        return result
