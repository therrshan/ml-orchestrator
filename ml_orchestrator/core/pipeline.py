"""
Main Pipeline class for orchestrating task execution.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .task import Task, TaskStatus
from .dag import DAG
from .state import StateManager, PipelineStatus
from ..config.parser import ConfigParser
from ..config.validator import ConfigValidator
from ..utils.exceptions import PipelineError, ExecutionError, ConfigurationError
from ..utils.logging import get_logger


class Pipeline:
    """
    Main pipeline orchestrator for managing task execution workflow.
    """
    
    def __init__(self, name: str, tasks: List[Task], metadata: Dict[str, Any] = None,
                 state_dir: str = ".ml_orchestrator"):
        """
        Initialize pipeline with tasks.
        
        Args:
            name: Pipeline name
            tasks: List of Task objects
            metadata: Optional pipeline metadata
            state_dir: Directory for state persistence
        """
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self.metadata = metadata or {}
        self.logger = get_logger(f"ml_orchestrator.pipeline.{name}")
        
        # Build DAG for dependency management
        self.dag = DAG(tasks)
        
        # State management
        self.state_manager = StateManager(state_dir)
        self.status = PipelineStatus.PENDING
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Execution control
        self._should_stop = False
        self._is_paused = False
    
    @classmethod
    def from_config(cls, config_path: str, pipeline_name: str = None, 
                   state_dir: str = ".ml_orchestrator") -> 'Pipeline':
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration file
            pipeline_name: Override pipeline name
            state_dir: Directory for state persistence
            
        Returns:
            Pipeline instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        parser = ConfigParser()
        validator = ConfigValidator()
        
        # Parse configuration
        config = parser.parse_file(config_path)
        tasks = parser.create_tasks(config)
        metadata = parser.get_pipeline_metadata(config)
        
        # Validate configuration
        validator.validate_full_config(config, tasks)
        
        # Use provided name or extract from config
        name = pipeline_name or metadata.get('name', Path(config_path).stem)
        
        return cls(name, tasks, metadata, state_dir)
    
    @classmethod
    def from_state(cls, pipeline_name: str, state_dir: str = ".ml_orchestrator") -> 'Pipeline':
        """
        Restore pipeline from saved state.
        
        Args:
            pipeline_name: Name of the pipeline to restore
            state_dir: Directory containing state files
            
        Returns:
            Restored Pipeline instance
            
        Raises:
            PipelineError: If state cannot be loaded
        """
        state_manager = StateManager(state_dir)
        pipeline_state = state_manager.load_pipeline_state(pipeline_name)
        
        if not pipeline_state:
            raise PipelineError(f"No saved state found for pipeline '{pipeline_name}'")
        
        # Restore tasks
        tasks = state_manager.restore_tasks_from_state(pipeline_state)
        metadata = pipeline_state.get('metadata', {})
        
        # Create pipeline
        pipeline = cls(pipeline_name, tasks, metadata, state_dir)
        
        # Restore pipeline state
        pipeline.status = PipelineStatus(pipeline_state.get('status', 'pending'))
        pipeline.completed_tasks = set(pipeline_state.get('completed_tasks', []))
        pipeline.failed_tasks = set(pipeline_state.get('failed_tasks', []))
        
        return pipeline
    
    def run(self, resume: bool = False) -> bool:
        """
        Execute the pipeline.
        
        Args:
            resume: Whether to resume from saved state
            
        Returns:
            True if pipeline completed successfully
        """
        try:
            # Handle resume logic
            if resume:
                self._resume_from_state()
            else:
                self._initialize_fresh_run()
            
            self.logger.info(f"Starting pipeline execution: {self.name}")
            self.status = PipelineStatus.RUNNING
            self.start_time = datetime.now()
            
            # Save initial state
            self._save_current_state()
            
            # Main execution loop
            success = self._execute_pipeline()
            
            # Final status update
            self.end_time = datetime.now()
            if success and not self._should_stop:
                self.status = PipelineStatus.SUCCESS
                self.logger.info(f"Pipeline completed successfully in {self._get_execution_time():.2f}s")
            elif self._should_stop:
                self.status = PipelineStatus.CANCELLED
                self.logger.info("Pipeline execution cancelled")
            else:
                self.status = PipelineStatus.FAILED
                self.logger.error("Pipeline execution failed")
            
            # Save final state
            self._save_current_state()
            
            return success
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = datetime.now()
            self.logger.error(f"Pipeline execution failed with error: {e}")
            self._save_current_state()
            raise PipelineError(f"Pipeline execution failed: {e}")
    
    def _initialize_fresh_run(self):
        """Initialize for a fresh pipeline run."""
        # Reset all task states
        for task in self.tasks.values():
            task.reset()
        
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.status = PipelineStatus.PENDING
        self._should_stop = False
        self._is_paused = False
    
    def _resume_from_state(self):
        """Resume pipeline from saved state."""
        saved_state = self.state_manager.load_pipeline_state(self.name)
        if saved_state:
            self.logger.info("Resuming pipeline from saved state")
            # State is already restored in from_state method
        else:
            self.logger.info("No saved state found, starting fresh")
            self._initialize_fresh_run()
    
    def _execute_pipeline(self) -> bool:
        """
        Main pipeline execution logic.
        
        Returns:
            True if all tasks completed successfully
        """
        total_tasks = len(self.tasks)
        
        while not self._should_stop and not self._is_paused:
            # Get tasks ready to run
            ready_tasks = self.dag.get_ready_tasks(self.completed_tasks)
            
            # Filter out already completed or failed tasks
            runnable_tasks = [
                task for task in ready_tasks
                if task.status in [TaskStatus.PENDING, TaskStatus.FAILED]
            ]
            
            if not runnable_tasks:
                # Check if we're done
                if len(self.completed_tasks) == total_tasks:
                    return True  # All tasks completed
                elif self.failed_tasks:
                    return False  # Some tasks failed and can't proceed
                else:
                    # No runnable tasks but not done - might be paused or waiting
                    time.sleep(1)
                    continue
            
            # Execute ready tasks
            for task in runnable_tasks:
                if self._should_stop:
                    break
                
                try:
                    self._execute_task(task)
                except ExecutionError as e:
                    self.logger.error(f"Task execution failed: {e}")
                    # Continue to next task - error handling is done in _execute_task
            
            # Brief pause between execution cycles
            time.sleep(0.1)
        
        return len(self.completed_tasks) == total_tasks
    
    def _execute_task(self, task: Task):
        """
        Execute a single task with proper error handling and state management.
        
        Args:
            task: Task to execute
        """
        self.logger.info(f"Executing task: {task.name}")
        
        # Add execution event
        self._add_execution_event("task_started", task.name)
        
        try:
            # Execute task with retry logic
            result = task.run_with_retry()
            
            # Task succeeded
            self.completed_tasks.add(task.name)
            self.failed_tasks.discard(task.name)
            
            self.logger.info(f"Task '{task.name}' completed successfully")
            self._add_execution_event("task_completed", task.name, {
                'execution_time': result.execution_time,
                'exit_code': result.exit_code
            })
            
        except ExecutionError as e:
            # Task failed
            self.failed_tasks.add(task.name)
            self.completed_tasks.discard(task.name)
            
            self.logger.error(f"Task '{task.name}' failed: {e}")
            self._add_execution_event("task_failed", task.name, {
                'error': str(e),
                'exit_code': getattr(e, 'exit_code', None)
            })
            
            # Re-raise to be handled by caller
            raise
        
        finally:
            # Always save state after task execution
            self._save_current_state()
    
    def pause(self):
        """Pause pipeline execution."""
        self._is_paused = True
        self.status = PipelineStatus.PAUSED
        self.logger.info("Pipeline execution paused")
        self._save_current_state()
    
    def resume(self):
        """Resume paused pipeline execution."""
        if self.status == PipelineStatus.PAUSED:
            self._is_paused = False
            self.status = PipelineStatus.RUNNING
            self.logger.info("Pipeline execution resumed")
    
    def stop(self):
        """Stop pipeline execution."""
        self._should_stop = True
        self.logger.info("Pipeline stop requested")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        total_tasks = len(self.tasks)
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        
        return {
            'name': self.name,
            'status': self.status.value,
            'total_tasks': total_tasks,
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'pending_tasks': total_tasks - completed_count - failed_count,
            'progress_percent': (completed_count / total_tasks) * 100 if total_tasks > 0 else 0,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self._get_execution_time(),
            'can_resume': self.status in [PipelineStatus.PAUSED, PipelineStatus.FAILED],
        }
    
    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task status dictionary or None if task not found
        """
        if task_name not in self.tasks:
            return None
        
        task = self.tasks[task_name]
        
        return {
            'name': task.name,
            'status': task.status.value,
            'command': task.command,
            'depends_on': task.depends_on,
            'retry_count': task.retry_count,
            'current_attempt': task.current_attempt,
            'error_message': task.error_message,
            'result': {
                'exit_code': task.result.exit_code,
                'execution_time': task.result.execution_time,
                'start_time': task.result.start_time.isoformat(),
                'end_time': task.result.end_time.isoformat(),
            } if task.result else None
        }
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        List all tasks with their current status.
        
        Returns:
            List of task status dictionaries
        """
        return [self.get_task_status(task_name) for task_name in self.tasks.keys()]
    
    def get_execution_plan(self) -> List[List[str]]:
        """
        Get the execution plan showing task batches.
        
        Returns:
            List of task name lists that can run in parallel
        """
        return self.dag.get_execution_plan()
    
    def _save_current_state(self):
        """Save current pipeline state to disk."""
        try:
            pipeline_state = self.state_manager.create_pipeline_state(
                self.name, list(self.tasks.values()), self.metadata
            )
            
            # Update with current state
            pipeline_state['status'] = self.status.value
            pipeline_state['completed_tasks'] = list(self.completed_tasks)
            pipeline_state['failed_tasks'] = list(self.failed_tasks)
            
            self.state_manager.save_pipeline_state(self.name, pipeline_state)
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
    
    def _add_execution_event(self, event_type: str, task_name: str = None, details: Dict[str, Any] = None):
        """Add an execution event to the pipeline history."""
        # For now, just log the event - we could extend this to save to state
        event_details = details or {}
        if task_name:
            self.logger.debug(f"Event: {event_type} for task {task_name} - {event_details}")
        else:
            self.logger.debug(f"Event: {event_type} - {event_details}")
    
    def _get_execution_time(self) -> float:
        """Get total execution time in seconds."""
        if not self.start_time:
            return 0.0
        
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def validate(self) -> List[str]:
        """
        Validate pipeline configuration.
        
        Returns:
            List of validation warnings
        """
        validator = ConfigValidator()
        return validator.get_validation_warnings(list(self.tasks.values()))
    
    def cleanup_state(self):
        """Clean up saved state for this pipeline."""
        self.state_manager.delete_pipeline_state(self.name)
        self.logger.info(f"Cleaned up state for pipeline '{self.name}'")