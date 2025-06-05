"""
Custom exceptions for the ML orchestrator package.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""
    pass


class PipelineError(OrchestratorError):
    """Raised when pipeline-level errors occur."""
    pass


class TaskError(OrchestratorError):
    """Raised when task-level errors occur."""
    pass


class ConfigurationError(OrchestratorError):
    """Raised when configuration parsing or validation fails."""
    pass


class DAGError(OrchestratorError):
    """Raised when DAG construction or validation fails."""
    pass


class StateError(OrchestratorError):
    """Raised when state persistence operations fail."""
    pass


class ExecutionError(TaskError):
    """Raised when task execution fails."""
    
    def __init__(self, message, task_name=None, exit_code=None, stdout=None, stderr=None):
        super().__init__(message)
        self.task_name = task_name
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
    
    def __str__(self):
        msg = super().__str__()
        if self.task_name:
            msg = f"Task '{self.task_name}': {msg}"
        if self.exit_code is not None:
            msg += f" (exit code: {self.exit_code})"
        return msg


class DependencyError(DAGError):
    """Raised when task dependencies are invalid."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependencies are detected in the DAG."""
    pass