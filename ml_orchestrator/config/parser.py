"""
Configuration file parsing for pipeline definitions.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import re

from ..core.task import Task
from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger


class ConfigParser:
    """Parser for pipeline configuration files."""
    
    def __init__(self):
        self.logger = get_logger("ml_orchestrator.config")
    
    def parse_file(self, config_path: str) -> Dict[str, Any]:
        """
        Parse configuration file (YAML or JSON).
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            ConfigurationError: If parsing fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse based on file extension
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(content)
            elif config_path.suffix.lower() == '.json':
                config = json.loads(content)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )
            
            self.logger.info(f"Successfully parsed configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON parsing error: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Supports formats:
        - ${VAR_NAME}
        - ${VAR_NAME:-default_value}
        - $VAR_NAME
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables substituted
        """
        # Pattern for ${VAR_NAME} and ${VAR_NAME:-default}
        def replace_env_var(match):
            var_expr = match.group(1)
            
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                value = os.getenv(var_expr)
                if value is None:
                    raise ConfigurationError(f"Environment variable '{var_expr}' not found")
                return value
        
        # Replace ${VAR} patterns
        content = re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
        
        # Replace $VAR patterns (simple form)
        def replace_simple_var(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ConfigurationError(f"Environment variable '{var_name}' not found")
            return value
        
        content = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_simple_var, content)
        
        return content
    
    def create_tasks(self, config: Dict[str, Any]) -> List[Task]:
        """
        Create Task objects from configuration.
        
        Args:
            config: Parsed configuration dictionary
            
        Returns:
            List of Task objects
            
        Raises:
            ConfigurationError: If task creation fails
        """
        if 'tasks' not in config:
            raise ConfigurationError("Configuration must contain 'tasks' section")
        
        tasks = []
        task_configs = config['tasks']
        
        if not isinstance(task_configs, list):
            raise ConfigurationError("'tasks' must be a list")
        
        for i, task_config in enumerate(task_configs):
            try:
                task = self._create_task_from_config(task_config)
                tasks.append(task)
            except Exception as e:
                raise ConfigurationError(f"Error creating task at index {i}: {e}")
        
        return tasks
    
    def _create_task_from_config(self, task_config: Dict[str, Any]) -> Task:
        """
        Create a Task object from configuration dictionary.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            Task object
            
        Raises:
            ConfigurationError: If required fields are missing or invalid
        """
        # Validate required fields
        if 'name' not in task_config:
            raise ConfigurationError("Task must have 'name' field")
        if 'command' not in task_config:
            raise ConfigurationError(f"Task '{task_config['name']}' must have 'command' field")
        
        # Extract task parameters
        name = task_config['name']
        command = task_config['command']
        depends_on = task_config.get('depends_on', [])
        retry_count = task_config.get('retry_count', 0)
        timeout = task_config.get('timeout')
        working_dir = task_config.get('working_dir')
        environment = task_config.get('environment', {})
        
        # Validate types
        if not isinstance(name, str):
            raise ConfigurationError("Task 'name' must be a string")
        if not isinstance(command, str):
            raise ConfigurationError(f"Task '{name}' 'command' must be a string")
        if not isinstance(depends_on, list):
            raise ConfigurationError(f"Task '{name}' 'depends_on' must be a list")
        if not isinstance(retry_count, int) or retry_count < 0:
            raise ConfigurationError(f"Task '{name}' 'retry_count' must be a non-negative integer")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ConfigurationError(f"Task '{name}' 'timeout' must be a positive integer")
        if working_dir is not None and not isinstance(working_dir, str):
            raise ConfigurationError(f"Task '{name}' 'working_dir' must be a string")
        if not isinstance(environment, dict):
            raise ConfigurationError(f"Task '{name}' 'environment' must be a dictionary")
        
        # Validate depends_on contains only strings
        for dep in depends_on:
            if not isinstance(dep, str):
                raise ConfigurationError(f"Task '{name}' dependency must be a string, got: {dep}")
        
        # Validate environment variables are strings
        for key, value in environment.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ConfigurationError(f"Task '{name}' environment variables must be strings")
        
        return Task(
            name=name,
            command=command,
            depends_on=depends_on,
            retry_count=retry_count,
            timeout=timeout,
            working_dir=working_dir,
            environment=environment
        )
    
    def get_pipeline_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pipeline metadata from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Pipeline metadata dictionary
        """
        metadata = {
            'name': config.get('name', 'unnamed_pipeline'),
            'description': config.get('description', ''),
            'version': config.get('version', '1.0'),
            'author': config.get('author', ''),
            'tags': config.get('tags', []),
        }
        
        return metadata
    
    def validate_task_names(self, tasks: List[Task]) -> None:
        """
        Validate that task names are unique and valid.
        
        Args:
            tasks: List of Task objects
            
        Raises:
            ConfigurationError: If validation fails
        """
        seen_names = set()
        
        for task in tasks:
            if not task.name:
                raise ConfigurationError("Task name cannot be empty")
            
            if task.name in seen_names:
                raise ConfigurationError(f"Duplicate task name: '{task.name}'")
            
            seen_names.add(task.name)
            
            # Validate name format (alphanumeric, underscore, hyphen)
            if not re.match(r'^[a-zA-Z0-9_-]+$', task.name):
                raise ConfigurationError(
                    f"Task name '{task.name}' contains invalid characters. "
                    "Only alphanumeric characters, underscores, and hyphens are allowed."
                )