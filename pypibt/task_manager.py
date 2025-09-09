"""
Task management system for PIBT with continuous task assignment.

This module handles task pools, assignment logic, and tracking of task completion
for multi-agent path finding scenarios where agents continuously get new tasks.
"""

from typing import Optional, Dict, List, Tuple
from .enums import Config, Coord


class TaskManager:
    """Manages task assignment and tracking for multi-agent systems."""
    
    def __init__(self, task_pool: List[Tuple[int, int, int]], num_agents: int, cycle_tasks: bool = True):
        """Initialize the task manager.
        
        Args:
            task_pool: List of tasks in format [(task_id, y, x), ...]
            num_agents: Number of agents in the system
            cycle_tasks: Whether to cycle through task pool when exhausted
            
        Raises:
            ValueError: If task pool is too small for the number of agents
        """
        # Validate task pool size
        if len(task_pool) < 2 * num_agents:
            raise ValueError(
                f"Task pool must be at least twice the number of agents. "
                f"Got {len(task_pool)} tasks for {num_agents} agents. "
                f"Minimum required: {2 * num_agents} tasks."
            )
        
        self.task_pool = task_pool.copy()
        self.num_agents = num_agents
        self.cycle_tasks = cycle_tasks
        
        # Current assignments: agent_id -> (internal_task_id, original_task_id, y, x)
        self.current_assignments: Dict[int, Tuple[int, int, int, int]] = {}
        
        # Task assignment history for logging
        self.assignment_history: List[Tuple[int, int, str, int]] = []  # (task_id, agent_id, action, timestep)
        
        # Track which task each agent should get next (round-robin with cycling)
        self.next_task_indices: List[int] = [i for i in range(num_agents)]
        
        # Completed tasks
        self.completed_tasks: List[Tuple[int, int, int]] = []
        
        # Track how many cycles we've completed
        self.cycle_count: int = 0
        
        # Initial assignment
        self._assign_initial_tasks()
    
    def _assign_initial_tasks(self, logger=None):
        """Assign initial tasks to all agents using round-robin."""
        for agent_id in range(self.num_agents):
            self._assign_next_task_to_agent(agent_id, 0, logger)
    
    def _assign_next_task_to_agent(self, agent_id: int, timestep: int = 0, logger=None) -> bool:
        """Assign the next task to a specific agent.
        
        Args:
            agent_id: ID of the agent to assign task to
            timestep: Current timestep
            logger: Optional logger to add assignment events and tasks to
            
        Returns:
            True if a task was assigned, False if no tasks available
        """
        if agent_id >= len(self.next_task_indices):
            return False
            
        task_index = self.next_task_indices[agent_id]
        
        # If we've exhausted the task pool and cycling is enabled, wrap around
        if task_index >= len(self.task_pool):
            if self.cycle_tasks:
                # Reset to start of pool with proper offset for this agent
                cycles_completed = task_index // len(self.task_pool)
                offset_in_cycle = task_index % len(self.task_pool)
                
                # If this is the first time we're cycling for this agent, update cycle count
                if offset_in_cycle < self.num_agents:
                    self.cycle_count = max(self.cycle_count, cycles_completed)
                
                task_index = offset_in_cycle
            else:
                return False
        
        if task_index < len(self.task_pool):
            original_task_id, y, x = self.task_pool[task_index]
            
            # For internal tracking, use a unique ID that includes cycle info
            cycles_for_this_task = self.next_task_indices[agent_id] // len(self.task_pool)
            internal_unique_id = original_task_id + (cycles_for_this_task * len(self.task_pool))
            
            # Store both internal and original task IDs for clean tracking
            self.current_assignments[agent_id] = (internal_unique_id, original_task_id, y, x)
            
            # Log the assignment using the ORIGINAL task ID (not inflated)
            self.assignment_history.append((original_task_id, agent_id, "assigned", timestep))
            
            # Add to logger if provided
            if logger:
                logger.add_event(original_task_id, agent_id, "assigned", timestep)
                logger.add_task(original_task_id, y, x)
            
            # Update the next task index for this agent
            self.next_task_indices[agent_id] += self.num_agents
            
            return True
        
        return False
    
    def get_current_goals(self) -> Config:
        """Get current goal positions for all agents.
        
        Returns:
            List of current goal positions [(y, x), ...]
        """
        goals = []
        for agent_id in range(self.num_agents):
            if agent_id in self.current_assignments:
                _, _, y, x = self.current_assignments[agent_id]
                goals.append((y, x))
            else:
                # No task assigned, use some default position or None
                goals.append((0, 0))  # Default position
        
        return goals
    
    def check_and_update_completed_tasks(self, current_positions: Config, timestep: int = 0, logger=None) -> List[int]:
        """Check if any agents have reached their goals and assign new tasks.
        
        Args:
            current_positions: Current positions of all agents [(y, x), ...]
            timestep: Current timestep
            logger: Optional logger to add task events and used tasks to
            
        Returns:
            List of agent IDs that completed tasks this timestep
        """
        completed_agents = []
        
        for agent_id in range(min(len(current_positions), self.num_agents)):
            if agent_id in self.current_assignments:
                internal_task_id, original_task_id, goal_y, goal_x = self.current_assignments[agent_id]
                agent_y, agent_x = current_positions[agent_id]
                
                # Check if agent reached the goal
                if agent_y == goal_y and agent_x == goal_x:
                    
                    # Mark task as completed with original ID
                    self.completed_tasks.append((original_task_id, goal_y, goal_x))
                    self.assignment_history.append((original_task_id, agent_id, "finished", timestep))
                    completed_agents.append(agent_id)
                    
                    # Add to logger if provided
                    if logger:
                        logger.add_event(original_task_id, agent_id, "finished", timestep)
                        logger.add_task(original_task_id, goal_y, goal_x)
                    
                    # Remove current assignment
                    del self.current_assignments[agent_id]
                    
                    # Assign new task if available
                    self._assign_next_task_to_agent(agent_id, timestep, logger)
        
        return completed_agents
    
    def get_task_for_agent(self, agent_id: int) -> Optional[Tuple[int, int, int]]:
        """Get the current task for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Current task (internal_task_id, y, x) or None if no task assigned
        """
        assignment = self.current_assignments.get(agent_id)
        if assignment:
            internal_task_id, original_task_id, y, x = assignment
            return (internal_task_id, y, x)  # Return internal ID for compatibility
        return None
    
    def get_assignment_events(self) -> List[List]:
        """Get all assignment events for logging.
        
        Returns:
            List of events in format [[task_id, agent_id, "action"], ...]
        """
        return [[task_id, agent_id, action, timestep] for task_id, agent_id, action, timestep in self.assignment_history]
    
    
    def log_initial_assignments(self, logger):
        """Log the initial task assignments that were made during initialization."""
        if logger:
            for agent_id, (internal_task_id, original_task_id, y, x) in self.current_assignments.items():
                logger.add_event(original_task_id, agent_id, "assigned", 0)  # Initial assignments at timestep 0
                logger.add_task(original_task_id, y, x)
    
    def get_all_tasks_for_logging(self) -> List[List]:
        """Get all tasks in logging format.
        
        Returns:
            List of tasks in format [[task_id, y, x], ...]
        """
        return [[task_id, y, x] for task_id, y, x in self.task_pool]
    
    def has_active_tasks(self) -> bool:
        """Check if there are any active or remaining tasks.
        
        Returns:
            True if there are active assignments or remaining tasks
        """
        if self.cycle_tasks:
            # With cycling enabled, we always have tasks available
            return len(self.current_assignments) > 0
        else:
            # Without cycling, check if there are remaining tasks in the pool
            return len(self.current_assignments) > 0 or any(
                idx < len(self.task_pool) for idx in self.next_task_indices
            )
    
    def get_stats(self) -> Dict:
        """Get statistics about task completion.
        
        Returns:
            Dictionary with task statistics
        """
        return {
            "total_tasks": len(self.task_pool),
            "completed_tasks": len(self.completed_tasks),
            "active_assignments": len(self.current_assignments),
            "total_events": len(self.assignment_history),
            "cycle_count": self.cycle_count,
            "cycling_enabled": self.cycle_tasks,
        }
