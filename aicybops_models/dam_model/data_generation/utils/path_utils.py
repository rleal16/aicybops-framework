"""
Utility functions for path resolution.

This module provides functions for resolving relative paths consistently
across data generation scripts.
"""

from pathlib import Path
from typing import Union


def resolve_path(path: Union[str, Path], 
                base_dir: Path, 
                must_exist: bool = False) -> Path:
    """
    Resolve a path, handling both absolute and relative paths.
    
    Args:
        path: Path to resolve (string or Path object)
        base_dir: Base directory for resolving relative paths (required)
        must_exist: If True, raises FileNotFoundError if path doesn't exist
    
    Returns:
        Resolved Path object
    
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    path = Path(path)
    
    # If absolute, return as-is
    if path.is_absolute():
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path
    
    # Resolve relative to base_dir
    base_dir = Path(base_dir)
    resolved_path = base_dir / path
    
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path not found: {resolved_path}")
    
    return resolved_path


def resolve_path_from_config(config_path: Path, 
                             path_key: str, 
                             config: dict,
                             must_exist: bool = False) -> Path:
    """
    Resolve a path from config file, relative to config file's directory.
    
    Args:
        config_path: Path to the config file
        path_key: Key in config dict containing the path (supports nested keys with '.')
        config: Config dictionary
        must_exist: If True, raises FileNotFoundError if path doesn't exist
    
    Returns:
        Resolved Path object
    
    Raises:
        ValueError: If path_key not found in config
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    # Handle nested keys (e.g., 'output_paths.generated_logs')
    keys = path_key.split('.')
    value = config
    for key in keys:
        if key not in value:
            raise ValueError(f"Config file must contain '{path_key}' field")
        value = value[key]
    
    path_str = value
    base_dir = config_path.parent
    
    return resolve_path(path_str, base_dir, must_exist)


def resolve_path_from_script_dir(path: Union[str, Path],
                                 script_file: str,
                                 must_exist: bool = False) -> Path:
    """
    Resolve a path relative to the script's directory.
    
    Args:
        path: Path to resolve (string or Path object)
        script_file: __file__ from the calling script (to determine script directory)
        must_exist: If True, raises FileNotFoundError if path doesn't exist
    
    Returns:
        Resolved Path object
    
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    script_dir = Path(script_file).parent
    return resolve_path(path, base_dir=script_dir, must_exist=must_exist)
