"""
ML Pipeline Orchestrator - A lightweight workflow management system for ML tasks.
"""

__version__ = "0.1.1"
__author__ = "Darshan Rajopadhye"
__email__ = "rajopadhye.d@northeastern.edu"

# Core imports
from .core.pipeline import Pipeline
from .core.task import Task, TaskStatus
from .core.dag import DAG
from .core.state import StateManager, PipelineStatus

# Configuration imports
from .config.parser import ConfigParser
from .config.validator import ConfigValidator

# Utility imports
from .utils.exceptions import (
    OrchestratorError,
    PipelineError,
    TaskError,
    ConfigurationError,
    DAGError,
    StateError,
    ExecutionError,
    DependencyError,
    CircularDependencyError
)
from .utils.logging import setup_logging, get_logger

# Convenience imports for common use cases
__all__ = [
    # Core classes
    "Pipeline",
    "Task",
    "TaskStatus",
    "DAG",
    "StateManager",
    "PipelineStatus",
    
    # Configuration
    "ConfigParser",
    "ConfigValidator",
    
    # Exceptions
    "OrchestratorError",
    "PipelineError",
    "TaskError",
    "ConfigurationError",
    "DAGError",
    "StateError",
    "ExecutionError",
    "DependencyError",
    "CircularDependencyError",
    
    # Utilities
    "setup_logging",
    "get_logger",
    
    # Version
    "__version__",
]


def create_pipeline_from_config(config_path: str, **kwargs) -> Pipeline:
    """
    Convenience function to create a pipeline from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        **kwargs: Additional arguments passed to Pipeline.from_config()
        
    Returns:
        Pipeline instance
    """
    return Pipeline.from_config(config_path, **kwargs)


def list_saved_pipelines(state_dir: str = ".ml_orchestrator") -> list:
    """
    Convenience function to list all saved pipeline states.
    
    Args:
        state_dir: Directory containing state files
        
    Returns:
        List of pipeline names with saved states
    """
    state_manager = StateManager(state_dir)
    return state_manager.list_pipeline_states()