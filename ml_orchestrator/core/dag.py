"""
Directed Acyclic Graph (DAG) operations for task dependency management.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

from .task import Task
from ..utils.exceptions import CircularDependencyError, DependencyError
from ..utils.logging import get_logger


class DAG:
    """
    Directed Acyclic Graph for managing task dependencies.
    """
    
    def __init__(self, tasks: List[Task]):
        """
        Initialize DAG with list of tasks.
        
        Args:
            tasks: List of Task objects
            
        Raises:
            DependencyError: If dependencies are invalid
            CircularDependencyError: If circular dependencies exist
        """
        self.logger = get_logger("ml_orchestrator.dag")
        self.tasks = {task.name: task for task in tasks}
        self.adjacency_list = defaultdict(list)  # task -> [dependent_tasks]
        self.reverse_adjacency = defaultdict(list)  # task -> [dependency_tasks]
        self.in_degree = defaultdict(int)  # task -> number of dependencies
        
        self._build_graph()
        self._validate_dependencies()
        self._detect_cycles()
    
    def _build_graph(self):
        """Build the adjacency lists and in-degree counts."""
        for task in self.tasks.values():
            # Initialize in-degree for all tasks
            self.in_degree[task.name] = len(task.depends_on)
            
            # Build adjacency lists
            for dependency in task.depends_on:
                self.adjacency_list[dependency].append(task.name)
                self.reverse_adjacency[task.name].append(dependency)
    
    def _validate_dependencies(self):
        """Validate that all dependencies exist."""
        for task in self.tasks.values():
            for dependency in task.depends_on:
                if dependency not in self.tasks:
                    raise DependencyError(
                        f"Task '{task.name}' depends on '{dependency}' which does not exist"
                    )
    
    def _detect_cycles(self):
        """
        Detect circular dependencies using DFS.
        
        Raises:
            CircularDependencyError: If cycles are found
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {task_name: WHITE for task_name in self.tasks}
        
        def dfs(node, path):
            if colors[node] == GRAY:
                # Found a back edge - cycle detected
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise CircularDependencyError(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )
            
            if colors[node] == BLACK:
                return
            
            colors[node] = GRAY
            path.append(node)
            
            for neighbor in self.adjacency_list[node]:
                dfs(neighbor, path)
            
            colors[node] = BLACK
            path.pop()
        
        for task_name in self.tasks:
            if colors[task_name] == WHITE:
                dfs(task_name, [])
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """
        Get tasks that are ready to run (all dependencies completed).
        
        Args:
            completed_tasks: Set of task names that have completed successfully
            
        Returns:
            List of tasks ready to execute
        """
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.can_run(completed_tasks):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order using Kahn's algorithm.
        
        Returns:
            List of task names in dependency order
        """
        # Copy in-degrees to avoid modifying original
        in_degree_copy = self.in_degree.copy()
        queue = deque()
        result = []
        
        # Find all tasks with no dependencies
        for task_name, degree in in_degree_copy.items():
            if degree == 0:
                queue.append(task_name)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree for all dependent tasks
            for dependent in self.adjacency_list[current]:
                in_degree_copy[dependent] -= 1
                if in_degree_copy[dependent] == 0:
                    queue.append(dependent)
        
        # Check if all tasks were processed (no cycles)
        if len(result) != len(self.tasks):
            remaining_tasks = set(self.tasks.keys()) - set(result)
            raise CircularDependencyError(
                f"Circular dependency prevents completion. Remaining tasks: {remaining_tasks}"
            )
        
        return result
    
    def get_task_levels(self) -> Dict[int, List[str]]:
        """
        Group tasks by execution level (tasks in same level can run in parallel).
        
        Returns:
            Dictionary mapping level number to list of task names
        """
        levels = defaultdict(list)
        task_levels = {}
        
        # Calculate level for each task
        def calculate_level(task_name):
            if task_name in task_levels:
                return task_levels[task_name]
            
            if not self.reverse_adjacency[task_name]:
                # No dependencies - level 0
                task_levels[task_name] = 0
            else:
                # Level is max of dependency levels + 1
                max_dep_level = max(
                    calculate_level(dep) for dep in self.reverse_adjacency[task_name]
                )
                task_levels[task_name] = max_dep_level + 1
            
            return task_levels[task_name]
        
        # Calculate level for all tasks
        for task_name in self.tasks:
            level = calculate_level(task_name)
            levels[level].append(task_name)
        
        return dict(levels)
    
    def get_dependencies(self, task_name: str) -> List[str]:
        """Get direct dependencies of a task."""
        if task_name not in self.tasks:
            raise DependencyError(f"Task '{task_name}' does not exist")
        return self.reverse_adjacency[task_name].copy()
    
    def get_dependents(self, task_name: str) -> List[str]:
        """Get direct dependents of a task."""
        if task_name not in self.tasks:
            raise DependencyError(f"Task '{task_name}' does not exist")
        return self.adjacency_list[task_name].copy()
    
    def get_all_dependencies(self, task_name: str) -> Set[str]:
        """Get all transitive dependencies of a task."""
        if task_name not in self.tasks:
            raise DependencyError(f"Task '{task_name}' does not exist")
        
        visited = set()
        stack = [task_name]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for dep in self.reverse_adjacency[current]:
                if dep not in visited:
                    stack.append(dep)
        
        # Remove the task itself from its dependencies
        visited.discard(task_name)
        return visited
    
    def validate_task_completion(self, task_name: str, completed_tasks: Set[str]) -> bool:
        """
        Check if a task can be marked as completed (all its dependencies are done).
        
        Args:
            task_name: Name of task to validate
            completed_tasks: Set of completed task names
            
        Returns:
            True if task can be completed
        """
        if task_name not in self.tasks:
            return False
        
        task_deps = set(self.reverse_adjacency[task_name])
        return task_deps.issubset(completed_tasks)
    
    def get_execution_plan(self) -> List[List[str]]:
        """
        Get execution plan as list of task batches that can run in parallel.
        
        Returns:
            List of lists, where inner lists contain tasks that can run in parallel
        """
        levels = self.get_task_levels()
        plan = []
        
        for level in sorted(levels.keys()):
            plan.append(levels[level])
        
        return plan