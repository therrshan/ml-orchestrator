"""
Configuration validation for pipeline definitions.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Set

from ..core.task import Task
from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger


class ConfigValidator:
    """Validator for pipeline configurations."""
    
    def __init__(self):
        self.logger = get_logger("ml_orchestrator.validator")
    
    def validate_full_config(self, config: Dict[str, Any], tasks: List[Task]) -> None:
        """
        Perform comprehensive validation of configuration and tasks.
        
        Args:
            config: Raw configuration dictionary
            tasks: List of parsed Task objects
            
        Raises:
            ConfigurationError: If validation fails
        """
        self.logger.info("Starting configuration validation")
        
        # Basic structure validation
        self._validate_config_structure(config)
        
        # Task validation
        self._validate_tasks(tasks)
        
        # Dependency validation
        self._validate_dependencies(tasks)
        
        # Command validation
        self._validate_commands(tasks)
        
        # Working directory validation
        self._validate_working_directories(tasks)
        
        self.logger.info("Configuration validation completed successfully")
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate basic configuration structure."""
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Required fields
        required_fields = ['tasks']
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field: '{field}'")
        
        # Optional fields with type validation
        optional_fields = {
            'name': str,
            'description': str,
            'version': str,
            'author': str,
            'tags': list,
        }
        
        for field, expected_type in optional_fields.items():
            if field in config and not isinstance(config[field], expected_type):
                raise ConfigurationError(f"Field '{field}' must be of type {expected_type.__name__}")
    
    def _validate_tasks(self, tasks: List[Task]) -> None:
        """Validate task list and individual tasks."""
        if not tasks:
            raise ConfigurationError("Pipeline must contain at least one task")
        
        if len(tasks) > 1000:  # Reasonable upper limit
            raise ConfigurationError("Pipeline cannot contain more than 1000 tasks")
        
        # Validate individual tasks
        for task in tasks:
            self._validate_single_task(task)
    
    def _validate_single_task(self, task: Task) -> None:
        """Validate a single task."""
        # Name validation
        if not task.name or not task.name.strip():
            raise ConfigurationError("Task name cannot be empty")
        
        if len(task.name) > 100:
            raise ConfigurationError(f"Task name '{task.name}' is too long (max 100 characters)")
        
        # Command validation
        if not task.command or not task.command.strip():
            raise ConfigurationError(f"Task '{task.name}' command cannot be empty")
        
        # Retry count validation
        if task.retry_count < 0 or task.retry_count > 10:
            raise ConfigurationError(f"Task '{task.name}' retry_count must be between 0 and 10")
        
        # Timeout validation
        if task.timeout is not None:
            if task.timeout <= 0 or task.timeout > 86400:  # Max 24 hours
                raise ConfigurationError(f"Task '{task.name}' timeout must be between 1 and 86400 seconds")
        
        # Environment variables validation
        for key, value in task.environment.items():
            if not key or not isinstance(key, str):
                raise ConfigurationError(f"Task '{task.name}' environment variable key must be a non-empty string")
            if not isinstance(value, str):
                raise ConfigurationError(f"Task '{task.name}' environment variable value must be a string")
    
    def _validate_dependencies(self, tasks: List[Task]) -> None:
        """Validate task dependencies."""
        task_names = {task.name for task in tasks}
        
        for task in tasks:
            # Check that all dependencies exist
            for dep in task.depends_on:
                if dep not in task_names:
                    raise ConfigurationError(f"Task '{task.name}' depends on non-existent task '{dep}'")
            
            # Check for self-dependency
            if task.name in task.depends_on:
                raise ConfigurationError(f"Task '{task.name}' cannot depend on itself")
            
            # Check for duplicate dependencies
            if len(task.depends_on) != len(set(task.depends_on)):
                duplicates = [dep for dep in task.depends_on if task.depends_on.count(dep) > 1]
                raise ConfigurationError(f"Task '{task.name}' has duplicate dependencies: {duplicates}")
    
    def _validate_commands(self, tasks: List[Task]) -> None:
        """Validate task commands."""
        for task in tasks:
            self._validate_command_accessibility(task)
    
    def _validate_command_accessibility(self, task: Task) -> None:
        """
        Validate that command components are accessible.
        
        Args:
            task: Task to validate
        """
        # Extract the main executable from the command
        command_parts = task.command.strip().split()
        if not command_parts:
            raise ConfigurationError(f"Task '{task.name}' has empty command")
        
        executable = command_parts[0]
        
        # Skip validation for shell built-ins and complex commands
        shell_builtins = {'cd', 'echo', 'export', 'set', 'unset', 'if', 'for', 'while', 'case'}
        if executable in shell_builtins or any(char in executable for char in ['|', '&', ';', '(', ')']):
            return
        
        # Check if it's a direct script path
        if executable.endswith('.py') or '/' in executable or '\\' in executable:
            self._validate_script_path(task, executable)
        else:
            # Check if executable is in PATH
            if not shutil.which(executable):
                self.logger.warning(f"Task '{task.name}': executable '{executable}' not found in PATH")
    
    def _validate_script_path(self, task: Task, script_path: str) -> None:
        """
        Validate that a script path exists and is accessible.
        
        Args:
            task: Task containing the script
            script_path: Path to the script
        """
        # Convert relative path based on working directory
        if task.working_dir and not os.path.isabs(script_path):
            full_path = Path(task.working_dir) / script_path
        else:
            full_path = Path(script_path)
        
        if not full_path.exists():
            self.logger.warning(f"Task '{task.name}': script '{script_path}' not found at {full_path}")
        elif not os.access(full_path, os.R_OK):
            raise ConfigurationError(f"Task '{task.name}': script '{script_path}' is not readable")
    
    def _validate_working_directories(self, tasks: List[Task]) -> None:
        """Validate working directories exist and are accessible."""
        for task in tasks:
            if task.working_dir:
                working_path = Path(task.working_dir)
                
                if not working_path.exists():
                    raise ConfigurationError(f"Task '{task.name}': working directory '{task.working_dir}' does not exist")
                
                if not working_path.is_dir():
                    raise ConfigurationError(f"Task '{task.name}': working directory '{task.working_dir}' is not a directory")
                
                if not os.access(working_path, os.R_OK | os.X_OK):
                    raise ConfigurationError(f"Task '{task.name}': working directory '{task.working_dir}' is not accessible")
    
    def validate_task_names_unique(self, tasks: List[Task]) -> None:
        """
        Validate that all task names are unique.
        
        Args:
            tasks: List of tasks to validate
            
        Raises:
            ConfigurationError: If duplicate names found
        """
        seen_names = set()
        duplicates = set()
        
        for task in tasks:
            if task.name in seen_names:
                duplicates.add(task.name)
            seen_names.add(task.name)
        
        if duplicates:
            raise ConfigurationError(f"Duplicate task names found: {', '.join(sorted(duplicates))}")
    
    def get_validation_warnings(self, tasks: List[Task]) -> List[str]:
        """
        Get non-critical validation warnings.
        
        Args:
            tasks: List of tasks to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        for task in tasks:
            # Check for very long commands
            if len(task.command) > 500:
                warnings.append(f"Task '{task.name}' has a very long command ({len(task.command)} characters)")
            
            # Check for high retry counts
            if task.retry_count > 5:
                warnings.append(f"Task '{task.name}' has high retry count ({task.retry_count})")
            
            # Check for very long timeouts
            if task.timeout and task.timeout > 3600:  # 1 hour
                warnings.append(f"Task '{task.name}' has long timeout ({task.timeout} seconds)")
            
            # Check for potential shell injection risks
            dangerous_chars = ['`', '$', '&&', '||', ';']
            if any(char in task.command for char in dangerous_chars):
                warnings.append(f"Task '{task.name}' command contains potentially dangerous shell characters")
        
        return warnings