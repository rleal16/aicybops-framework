import datetime
import random
import re
from string import Formatter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from .log_config_loader import LogConfigLoader



class LogGenerator:
    """
    A causal, template-based log generator that creates realistic microservice logs
    based on system metrics with proper temporal alignment for the DAM model.
    """

    
    def __init__(self, root_config_path: Path) -> None:
        """
        Initialize the log generator with root configuration file.
    
        Args:
            root_config_path: Path to root_config.json (required)
        
        Raises:
            ValueError: If root_config_path is None or validation fails
        """
        if root_config_path is None:
            raise ValueError("root_config_path is required")

        self.config_loader = LogConfigLoader(root_config_path)
        
        # Store loaded configurations for easy access
        self.templates = self.config_loader.templates
        self.rules = self.config_loader.rules
        self.contexts = self.config_loader.contexts
        
        # Track persistent state for hysteresis for state-driven logs
        # This is used in _state_driven_generation() to prevent alert spam.
        self.state_history = {}
        
        self.correlation_ids = {}
        
        # Cleanup configuration for state_history
        self._cleanup_counter = 0  # Counter for periodic cleanup
        self._cleanup_frequency = 50  # Clean up every N generations

        # Generation stats and tracking
        self.generation_stats = {
            'total_logs': 0,
            'templates_used': set(),
            'correlation_strength': 0.0,
            'template_diversity': 0.0,
        }
    
    
    def generate_log(self, metric_row: Dict[str, Any]) -> List[str]:
        """
        Generate a logs based for a given metric row.
        
        Args:
            metric_row: Dictionary containing metric data
        
        Returns:
            List of generated log strings with proper timestamps.
        """
        # extract base timestamp from metric row
        base_time = self._extract_timestamp(metric_row)
        service_name = metric_row.get('service_name', 'unknown-service')

        # initialize log collection
        all_logs = []

        # Generate event-driven logs
        event_driven_logs = self._event_driven_generation(metric_row, base_time)
        all_logs.extend(event_driven_logs)

        # Generate state-driven logs
        state_driven_logs = self._state_driven_generation(metric_row, base_time)
        all_logs.extend(state_driven_logs)

        if not all_logs:
            return None
        
        # Distribute timestamps within a 1-second window
        timestamped_logs = self._distribute_timestamps(all_logs, base_time)

        # update the generation stats
        self._update_generation_stats(timestamped_logs, metric_row)
        
        # Periodic cleanup of state_history to prevent memory growth
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_frequency:
            _ = self._cleanup_state_history(remove_inactive=True)
            self._cleanup_counter = 0  # Reset counter

        return timestamped_logs
    
    def _event_driven_generation(self, metric_row: Dict[str, Any], base_time: datetime.datetime) -> List[str]:
        """
        Generate event-driven logs based on the metric.
        These logs relflect the causes of the metrics values.

        Args:
            metric_row: Dictionary containing metric data
            base_time: Base datetime to start from

        Returns:
            List of event-driven log strings
        """
        logs = []
        service_name = metric_row.get('service_name', 'unknown-service')

        event_rules = self.rules.get('event_driven', {})

        # Process each metric that has event-driven rules
        for metric_name, value in metric_row.items():
            # skip non-metric fields
            if metric_name in ['service_name', 'timestamp']:
                continue
            
            # Skip non-numeric values (categorical fields like 'operation', 'device', etc.)
            if not isinstance(value, (int, float, np.number)):
                continue
            
            # Skip invalid numeric values (NaN, infinity, negative) - safety check
            if not np.isfinite(value) or value < 0:
                continue

            # check if metric has event-driven rules
            if metric_name in event_rules:
                rule = event_rules[metric_name]
                category = rule['category']
                subcategory = rule['subcategory']
                rate_factor = rule['rate_factor']
                min_logs = rule.get('min_logs', 0)
                max_logs = rule.get('max_logs', 10)

                # calculate the number of logs to generate using the poisson distribution
                expected_logs = value * rate_factor
                if expected_logs < 0 or not np.isfinite(expected_logs):
                    expected_logs = 0
                
                # The Poisson distribution models the log count as discrete, independent events.
                # It ensures the mean number of logs aligns with expected_logs (lambda) while generating realistic, non-deterministic variation around that mean.
                n_logs = max(min_logs, min(np.random.poisson(expected_logs), max_logs))

                # generate the logs for this metric
                for _ in range(n_logs):
                    template = self._select_template('event_driven', category, subcategory)
                    context = self._generate_context(
                        category, subcategory, template,
                        metric_name, value, service_name
                    )
                    log = self._populate_template(template, context)
                    logs.append(log)
        
        return logs

    def _select_template(self, template_type: str, category: str, subcategory: str) -> str:
        """
        Select a random template from the specified template type, category, and subcategory.
        
        Args:
            template_type: Template type ('event_driven' or 'state_driven')
            category: Template category (e.g., 'http_operations')
            subcategory: Template subcategory (e.g., 'request_received')
        
        Returns:
            Selected template string with placeholders
        
        Raises:
            KeyError: If template_type, category, or subcategory not found in templates
            ValueError: If no templates available for the specified combination
        """
        
        if template_type not in self.templates:
            raise KeyError(
                f"Template type '{template_type}' not found. "
                f"Available types: {list(self.templates.keys())}"
            )
        
        templates = self.templates[template_type]
        
        if category not in templates:
            raise KeyError(
                f"Template category '{category}' not found in template type '{template_type}'. "
                f"Available categories: {list(templates.keys())}"
            )
        
        if subcategory not in templates[category]:
            raise KeyError(
                f"Template subcategory '{subcategory}' not found in category '{category}' "
                f"(template type: '{template_type}'). "
                f"Available subcategories: {list(templates[category].keys())}"
            )
        
        template_list = templates[category][subcategory]
        
        if not template_list:
            raise ValueError(f"No templates available for {template_type}.{category}.{subcategory}")
        
        return random.choice(template_list)
    

    def _populate_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Populate template with context values to create a log message.

        Args:
            template: Template string with placeholders (e.g., "{timestamp} {level} {pid})
            context: Dictionary containing context values to populate the template
        
        Returns:
            Populated log message string
        """

        try:
            # use string formatting to replace the placeholders with the context values
            log_entry = template.format(**context)
            return log_entry
        except KeyError as e:
            missing_key = e.args[0]
            raise KeyError(f"Missing key {missing_key} in context for template: {template}") from e
        except Exception as e:
            raise ValueError(f"Failed to format template: '{template}' with context: {context}: {e}") from e

    def _extract_placeholders(self, template: str) -> List[str]:
        """
        Extract placeholders from a template string using the standard formatter parser.
        Returns a list of placeholders.
        Raises ValueError if the template is invalid.
        """
        try:
            placeholders = []
            for _lit, field_name, format_spec, _conv in Formatter().parse(template):
                if field_name:
                    if format_spec:
                        raise ValueError(f"Format specifiers are not allowed in templates: '{{{field_name}:{format_spec}}}'")
                    placeholders.append(field_name)
            return placeholders
        except ValueError as e:
            raise ValueError(f"Failed to extract placeholders from template: '{template}': {e}") from e

    def _generate_context(self, category: str, subcategory: str, template: str, 
                         metric_name: str, value: float, service_name: str, 
                        level: str = 'INFO', threshold: float = None) -> Dict[str, Any]:
        """
        Generate context data dynamically based on template placeholders and contexts.json.
        
        Args:
            category: Template category
            subcategory: Template subcategory
            template: The selected template string
            metric_name: Name of the metric that triggered this log
            value: Current value of the metric
            service_name: Name of the service
            level: Log level (from state-driven rules or default 'INFO')
            threshold: Threshold value from rule (for state-driven logs only, e.g., for "{metric} threshold exceeded: {value}% (threshold: {threshold}%)")
        
        Returns:
            Dictionary containing context data for all placeholders
        """
        # Extract placeholders from template
        placeholders = self._extract_placeholders(template)
        if not placeholders:
            raise ValueError(f"No valid placeholders found in template: '{template}'")

        # Placeholders that are dynamically replaced from metric data
        metrics_placeholders = {
            'timestamp': '{timestamp}', # will be replaced later in _distribute_timestamps
            'metric': metric_name,
            'value': value,
            'service_name': service_name,
            'threshold': threshold, # from state-driven rule (only used when template contains {threshold})
            'level': level,
        }
        
        context = {}
        
        # Populate each placeholder from contexts - call _get_context_value for all placeholders
        for placeholder in placeholders:
            if placeholder in metrics_placeholders:
                context[placeholder] = metrics_placeholders[placeholder]
            else:
                context[placeholder] = self._get_context_value(placeholder, category)
        
        return context

    def _get_context_value(self, placeholder: str, category: str) -> Any:
        """
        Get a random value for a placeholder from contexts.json.
        
        Args:
            placeholder: The placeholder name (e.g., 'method', 'endpoint')
            category: The template category to search in
        
        Returns:
            A random value from the context data
        
        Raises:
            KeyError: If placeholder not found in any context category
        """
        # First try the specific category
        if category in self.contexts and placeholder in self.contexts[category]:
            values = self.contexts[category][placeholder]
            if isinstance(values, list):
                return random.choice(values)
            elif isinstance(values, dict):
                # For nested dicts like normal_range, return a random value
                return random.choice(list(values.values()))
        
        # Then try common_log_fields (TODO: delete this in the future)
        if 'common_log_fields' in self.contexts and placeholder in self.contexts['common_log_fields']:
            values = self.contexts['common_log_fields'][placeholder]
            if isinstance(values, list):
                return random.choice(values)
        
        # Search all other categories (TODO: delete this in the future)
        for cat_name, cat_data in self.contexts.items():
            if placeholder in cat_data:
                values = cat_data[placeholder]
                if isinstance(values, list):
                    return random.choice(values)
                elif isinstance(values, dict):
                    return random.choice(list(values.values()))
        
        raise KeyError(f"Placeholder '{placeholder}' not found in contexts")
    
    def _threshold_crossed(self, value: float, threshold: float) -> bool:
        """
        Check if the threshold is crossed.
        """
        # We assume that the value is always non-negative, 
        # so when we want to trigger a log entry when the value is less than the threshold, we need to use a negative threshold in the rules.
        # As such, we need to check if the negative value is greater than the threshold.
        if threshold < 0:
            return -value >= threshold
        else:
            # When threshold is positive, we can just check if the value is greater than the threshold
            return value >= threshold

    def _state_driven_generation(self, metric_row: Dict[str, Any], base_time: datetime.datetime) -> List[str]:
        """
        Generate state-driven logs based on metric values crossing thresholds.
        Implements hysteresis to prevent alert spam.
        """
        logs = []
        service_name = metric_row.get('service_name', 'unknown-service')

        state_rules = self.rules.get('state_driven', {})
        
        # Process each metric that has state-driven rules
        for metric_name, value in metric_row.items():
            if metric_name in ['service_name', 'timestamp']:
                continue
            
            # Skip non-numeric values (categorical fields like 'operation', 'device', etc.)
            if not isinstance(value, (int, float, np.number)):
                continue
            
            # Skip invalid numeric values (NaN, infinity, negative) - safety check
            if not np.isfinite(value) or value < 0:
                continue
            
            if metric_name in state_rules:
                for state_name, rule in state_rules[metric_name].items():
                    threshold = rule['threshold']
                    level = rule['level']
                    category = rule['category']
                    subcategory = rule['subcategory']
                    # Note: persistence_steps and cooldown_steps are used to prevent false positives and alert spam and, together, represent the hysteresis -- the resistance to (rapid) changes.
                    persistence_steps = rule['persistence_steps'] # prevents false positives by not triggering a log entry every step that the threshold is crossed
                    cooldown_steps = rule['cooldown_steps'] # prevents alert spam by not triggering a log entry for the same threshold crossing too quickly

                    if self._threshold_crossed(value, threshold):
                        state_key = f"{service_name}:{metric_name}:{state_name}"
                        
                        # Update state history
                        if state_key not in self.state_history:
                            self.state_history[state_key] = {
                                'steps_active': 0,
                                'cooldown_remaining': 0
                            }
                        
                        state = self.state_history[state_key]

                        if state['cooldown_remaining'] > 0:
                            state['cooldown_remaining'] -= 1
                            continue
                        
                        # Increment active steps for every step that the threshold is crossed until persistence_steps is met to trigger a log entry generation
                        state['steps_active'] += 1
                        
                        # Generate log if persistence_steps is reached to trigger a log entry generation
                        if state['steps_active'] >= persistence_steps:
                            template = self._select_template('state_driven', category, subcategory)
                            context = self._generate_context(
                                category, subcategory, template,
                                metric_name, value, service_name, level, threshold
                            )
                            
                            log = self._populate_template(template, context)
                            logs.append(log)
                            
                            # Reset and start cooldown for cooldown_steps steps
                            state['steps_active'] = 0
                            state['cooldown_remaining'] = cooldown_steps
                    else:
                        # Threshold not crossed, reset state
                        state_key = f"{service_name}:{metric_name}:{state_name}"
                        if state_key in self.state_history:
                            self.state_history[state_key]['steps_active'] = 0
        
        return logs
    
    def _cleanup_state_history(self, remove_inactive: bool = True, max_size: Optional[int] = None) -> int:
        """
        Clean up stale entries from state_history to prevent memory growth.
        
        Args:
            remove_inactive: If True, remove entries where cooldown=0 and steps_active=0
            max_size: If provided, remove oldest entries when state_history exceeds this size
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        if remove_inactive:
            # Remove entries that are completely inactive (no cooldown, no active steps)
            inactive_keys = [
                key for key, state in self.state_history.items()
                if state['cooldown_remaining'] == 0 and state['steps_active'] == 0
            ]
            for key in inactive_keys:
                del self.state_history[key]
            removed_count += len(inactive_keys)
        
        # If state_history is still too large, remove oldest entries
        # (Note: Since dicts in Python 3.7+ maintain insertion order, we can remove first N)
        if max_size is not None and len(self.state_history) > max_size:
            keys_to_remove = list(self.state_history.keys())[:len(self.state_history) - max_size]
            for key in keys_to_remove:
                del self.state_history[key]
            removed_count += len(keys_to_remove)
        
        return removed_count
    
    def _update_generation_stats(self, logs: List[str], metric_row: Dict[str, Any]) -> None:
        """
        Update generation stats.
        
        # TODO: Track templates_used, compute correlation_strength, and template_diversity.
        """
        self.generation_stats['total_logs'] += len(logs)
                
    
    def _extract_timestamp(self, metric_row: Dict[str, Any]) -> datetime.datetime:
        
        """
        Extract and format timestamp from metric row.
        
        Args:
            metric_row: Dictionary containing metric data with 'timestamp' key

        Returns:
            datetime object representing the timestamp
        """
        if 'timestamp' not in metric_row:
            raise KeyError("Timestamp is required")
        
        ts = metric_row.get('timestamp')
        
        if ts is None:
            raise KeyError("Timestamp cannot be None")

        # if timestamp is already a datetime object, return it
        if isinstance(ts, datetime.datetime):
            return ts

        if isinstance(ts, str):
            try:
                # ISO format first
                return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        return datetime.datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
        
        if isinstance(ts, (int, float)):
            return datetime.datetime.fromtimestamp(ts)
        return datetime.datetime.now()

    def _calculate_offset_ms(self, i: int, total_logs: int) -> int:
        """
        Calculate the offset in milliseconds to distribute the logs within a 1-second window.
        """
        if total_logs > 1:
            # evenly distribute the logs within a 1-second window if there are more than one log/template, ensures the last log is at (or close to) the end of the window
            return int((i*1000.0) / (total_logs - 1)) 
        return 0

    def _distribute_timestamps(self, logs: List[str], base_time: datetime.datetime) -> List[str]:
        """
        Distribute logs within a 1-second window using millisecond offsets.
        
        Args:
            logs: List of log strings without timestamps
            base_time: Base datetime to start from
        
        Returns:
            List of log strings with distributed timestamps
        """
        if not logs:
            return []
        
        timestamped_logs = []
        total_logs = len(logs)

        for i, log in enumerate(logs):
            # Creates a timestamp offset to distribute the logs within a 1-second window
            offset_ms = self._calculate_offset_ms(i, total_logs)
            random_offset = random.randint(0, 50) # add some noise to the timestamps to prevent (unrealistic) perfect uniforme timestamps
            
            total_offset = min(offset_ms + random_offset, 999) # ensure the resulting timestamp is within the 1-second window
            
            # Create the actual timestamp by adding the offset to the base time
            log_time = base_time + datetime.timedelta(milliseconds=total_offset)
            ms = int(log_time.microsecond / 1000)
            timestamp_str = log_time.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
            
            # Replace the template's placeholder '{timestamp}' with the actual timestamp
            timestamped_log = log.replace("{timestamp}", timestamp_str)
            timestamped_logs.append(timestamped_log)
        return timestamped_logs



    