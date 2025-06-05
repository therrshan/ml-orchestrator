"""
Task representation and execution logic.
"""

import subprocess
import time
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..utils.exceptions import ExecutionError, TaskError
from ..utils.logging import get_logger


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of task execution."""
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    start_time: datetime
    end_time: datetime


@dataclass
class Task:
    """
    Represents a single task in the pipeline.
    """
    name: str
    command: str
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: Optional[int] = None
    working_dir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    current_attempt: int = 0
    result: Optional[TaskResult] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize task after creation."""
        self.logger = get_logger(f"ml_orchestrator.task.{self.name}")
    
    def can_run(self, completed_tasks: set) -> bool:
        """
        Check if this task can run based on its dependencies.
        
        Args:
            completed_tasks: Set of task names that have completed successfully
            
        Returns:
            True if all dependencies are satisfied
        """
        if self.status in [TaskStatus.RUNNING, TaskStatus.SUCCESS]:
            return False
        
        # Check if all dependencies are completed
        for dep in self.depends_on:
            if dep not in completed_tasks:
                return False
        
        return True
    
    def execute(self) -> TaskResult:
        """
        Execute the task command.
        
        Returns:
            TaskResult with execution details
            
        Raises:
            ExecutionError: If task execution fails
        """
        self.logger.info(f"Starting task execution (attempt {self.current_attempt + 1})")
        self.status = TaskStatus.RUNNING
        
        start_time = datetime.now()
        
        try:
            # Prepare environment
            env = None
            if self.environment:
                import os
                env = os.environ.copy()
                env.update(self.environment)
            
            # Execute command
            process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise ExecutionError(
                    f"Task timed out after {self.timeout} seconds",
                    task_name=self.name,
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr
                )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = TaskResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            if exit_code == 0:
                self.status = TaskStatus.SUCCESS
                self.result = result
                self.logger.info(f"Task completed successfully in {execution_time:.2f}s")
                return result
            else:
                error_msg = f"Command failed with exit code {exit_code}"
                if stderr:
                    error_msg += f"\nStderr: {stderr}"
                
                raise ExecutionError(
                    error_msg,
                    task_name=self.name,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr
                )
        
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.status = TaskStatus.FAILED
            self.error_message = str(e)
            self.logger.error(f"Task failed after {execution_time:.2f}s: {e}")
            raise
    
    def run_with_retry(self) -> TaskResult:
        """
        Execute task with retry logic.
        
        Returns:
            TaskResult if successful
            
        Raises:
            ExecutionError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.retry_count + 1):
            self.current_attempt = attempt
            
            try:
                return self.execute()
            except ExecutionError as e:
                last_error = e
                if attempt < self.retry_count:
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    self.status = TaskStatus.PENDING
                    time.sleep(1)  # Simple backoff
                else:
                    self.logger.error(f"All {self.retry_count + 1} attempts failed")
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise ExecutionError("Task execution failed", task_name=self.name)
    
    def reset(self):
        """Reset task state for re-execution."""
        self.status = TaskStatus.PENDING
        self.current_attempt = 0
        self.result = None
        self.error_message = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'name': self.name,
            'command': self.command,
            'depends_on': self.depends_on,
            'retry_count': self.retry_count,
            'timeout': self.timeout,
            'working_dir': self.working_dir,
            'environment': self.environment,
            'status': self.status.value,
            'current_attempt': self.current_attempt,
            'error_message': self.error_message,
            'result': {
                'exit_code': self.result.exit_code,
                'stdout': self.result.stdout,
                'stderr': self.result.stderr,
                'execution_time': self.result.execution_time,
                'start_time': self.result.start_time.isoformat(),
                'end_time': self.result.end_time.isoformat()
            } if self.result else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        task = cls(
            name=data['name'],
            command=data['command'],
            depends_on=data.get('depends_on', []),
            retry_count=data.get('retry_count', 0),
            timeout=data.get('timeout'),
            working_dir=data.get('working_dir'),
            environment=data.get('environment', {})
        )
        
        task.status = TaskStatus(data.get('status', 'pending'))
        task.current_attempt = data.get('current_attempt', 0)
        task.error_message = data.get('error_message')
        
        # Restore result if present
        if data.get('result'):
            result_data = data['result']
            task.result = TaskResult(
                exit_code=result_data['exit_code'],
                stdout=result_data['stdout'],
                stderr=result_data['stderr'],
                execution_time=result_data['execution_time'],
                start_time=datetime.fromisoformat(result_data['start_time']),
                end_time=datetime.fromisoformat(result_data['end_time'])
            )
        
        return task