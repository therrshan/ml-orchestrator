"""
State persistence and recovery for pipeline execution.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

from .task import Task, TaskStatus
from ..utils.exceptions import StateError
from ..utils.logging import get_logger


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StateManager:
    """Manages pipeline state persistence and recovery."""
    
    def __init__(self, state_dir: str = ".ml_orchestrator"):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.logger = get_logger("ml_orchestrator.state")
    
    def save_pipeline_state(self, pipeline_name: str, pipeline_state: Dict[str, Any]) -> None:
        """
        Save pipeline state to disk.
        
        Args:
            pipeline_name: Name of the pipeline
            pipeline_state: State dictionary to save
            
        Raises:
            StateError: If save operation fails
        """
        try:
            state_file = self.state_dir / f"{pipeline_name}.json"
            
            # Add timestamp to state
            pipeline_state['last_updated'] = datetime.now().isoformat()
            
            # Write to temporary file first, then rename for atomic operation
            temp_file = state_file.with_suffix('.json.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_state, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.rename(state_file)
            
            self.logger.debug(f"Saved pipeline state for '{pipeline_name}' to {state_file}")
            
        except Exception as e:
            raise StateError(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        Load pipeline state from disk.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Pipeline state dictionary or None if not found
            
        Raises:
            StateError: If load operation fails
        """
        try:
            state_file = self.state_dir / f"{pipeline_name}.json"
            
            if not state_file.exists():
                return None
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.logger.debug(f"Loaded pipeline state for '{pipeline_name}' from {state_file}")
            return state
            
        except json.JSONDecodeError as e:
            raise StateError(f"Invalid JSON in state file: {e}")
        except Exception as e:
            raise StateError(f"Failed to load pipeline state: {e}")
    
    def delete_pipeline_state(self, pipeline_name: str) -> bool:
        """
        Delete pipeline state file.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        try:
            state_file = self.state_dir / f"{pipeline_name}.json"
            
            if state_file.exists():
                state_file.unlink()
                self.logger.debug(f"Deleted pipeline state for '{pipeline_name}'")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete pipeline state: {e}")
            return False
    
    def list_pipeline_states(self) -> List[str]:
        """
        List all available pipeline states.
        
        Returns:
            List of pipeline names with saved states
        """
        try:
            pipeline_names = []
            
            for state_file in self.state_dir.glob("*.json"):
                if not state_file.name.endswith('.tmp'):
                    pipeline_name = state_file.stem
                    pipeline_names.append(pipeline_name)
            
            return sorted(pipeline_names)
            
        except Exception as e:
            self.logger.error(f"Failed to list pipeline states: {e}")
            return []
    
    def get_pipeline_info(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a pipeline state.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            Dictionary with pipeline info or None if not found
        """
        state = self.load_pipeline_state(pipeline_name)
        if not state:
            return None
        
        return {
            'name': state.get('name', pipeline_name),
            'status': state.get('status', 'unknown'),
            'created_at': state.get('created_at'),
            'last_updated': state.get('last_updated'),
            'total_tasks': len(state.get('tasks', {})),
            'completed_tasks': len([
                task for task in state.get('tasks', {}).values()
                if task.get('status') == TaskStatus.SUCCESS.value
            ]),
            'failed_tasks': len([
                task for task in state.get('tasks', {}).values()
                if task.get('status') == TaskStatus.FAILED.value
            ]),
        }
    
    def create_pipeline_state(self, pipeline_name: str, tasks: List[Task], 
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create initial pipeline state.
        
        Args:
            pipeline_name: Name of the pipeline
            tasks: List of Task objects
            metadata: Optional metadata dictionary
            
        Returns:
            Initial pipeline state dictionary
        """
        now = datetime.now().isoformat()
        
        state = {
            'name': pipeline_name,
            'status': PipelineStatus.PENDING.value,
            'created_at': now,
            'last_updated': now,
            'metadata': metadata or {},
            'tasks': {task.name: task.to_dict() for task in tasks},
            'execution_history': [],
            'completed_tasks': [],
            'failed_tasks': [],
        }
        
        return state
    
    def update_task_state(self, pipeline_state: Dict[str, Any], task: Task) -> None:
        """
        Update task state in pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            task: Task object with updated state
        """
        pipeline_state['tasks'][task.name] = task.to_dict()
        
        # Update completed/failed task lists
        completed_tasks = set(pipeline_state.get('completed_tasks', []))
        failed_tasks = set(pipeline_state.get('failed_tasks', []))
        
        if task.status == TaskStatus.SUCCESS:
            completed_tasks.add(task.name)
            failed_tasks.discard(task.name)
        elif task.status == TaskStatus.FAILED:
            failed_tasks.add(task.name)
            completed_tasks.discard(task.name)
        
        pipeline_state['completed_tasks'] = list(completed_tasks)
        pipeline_state['failed_tasks'] = list(failed_tasks)
    
    def add_execution_event(self, pipeline_state: Dict[str, Any], event_type: str, 
                          task_name: str = None, details: Dict[str, Any] = None) -> None:
        """
        Add execution event to pipeline history.
        
        Args:
            pipeline_state: Pipeline state dictionary
            event_type: Type of event (task_started, task_completed, etc.)
            task_name: Name of task related to event
            details: Additional event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'task_name': task_name,
            'details': details or {}
        }
        
        if 'execution_history' not in pipeline_state:
            pipeline_state['execution_history'] = []
        
        pipeline_state['execution_history'].append(event)
    
    def restore_tasks_from_state(self, pipeline_state: Dict[str, Any]) -> List[Task]:
        """
        Restore Task objects from pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            
        Returns:
            List of restored Task objects
        """
        tasks = []
        
        for task_data in pipeline_state.get('tasks', {}).values():
            task = Task.from_dict(task_data)
            tasks.append(task)
        
        return tasks
    
    def is_pipeline_recoverable(self, pipeline_name: str) -> bool:
        """
        Check if a pipeline can be recovered from its saved state.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            True if pipeline can be recovered
        """
        state = self.load_pipeline_state(pipeline_name)
        if not state:
            return False
        
        status = state.get('status')
        return status in [PipelineStatus.RUNNING.value, PipelineStatus.PAUSED.value, 
                         PipelineStatus.FAILED.value]
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        Clean up old pipeline state files.
        
        Args:
            max_age_days: Maximum age in days for state files
            
        Returns:
            Number of files cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            for state_file in self.state_dir.glob("*.json"):
                if state_file.stat().st_mtime < cutoff_time:
                    try:
                        state_file.unlink()
                        cleaned_count += 1
                        self.logger.debug(f"Cleaned up old state file: {state_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up {state_file}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old states: {e}")
            return 0
    
    def export_pipeline_state(self, pipeline_name: str, output_file: str) -> bool:
        """
        Export pipeline state to a file.
        
        Args:
            pipeline_name: Name of the pipeline
            output_file: Path to output file
            
        Returns:
            True if export successful
        """
        try:
            state = self.load_pipeline_state(pipeline_name)
            if not state:
                return False
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported pipeline state to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export pipeline state: {e}")
            return False
    
    def get_state_file_path(self, pipeline_name: str) -> Path:
        """Get the path to a pipeline's state file."""
        return self.state_dir / f"{pipeline_name}.json"