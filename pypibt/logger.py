"""
Logger module for saving PIBT planning results to JSON format.

This module provides functionality to log planning results including paths,
timings, and metadata in a structured JSON format suitable for analysis.
"""

import json
import time
from pathlib import Path
from typing import Optional, Union

from .enums import Action, Orientation, Config, OrientedConfig, Configs, OrientedConfigs
from .action_sequences import action_sequence_to_string


class PIBTLogger:
    """Logger for PIBT planning results."""
    
    def __init__(self, actionModel: str = "MAPF_T"):
        """Initialize the logger.
        
        Args:
            actionModel: The action model used (e.g., "MAPF_T", "PIBT", "MultiActionPIBT")
        """
        self.actionModel = actionModel
        self.start_time: Optional[float] = None
        self.planner_times: list[float] = []
        self.tasks: list[list] = []  # Format: [[id, y, x], ...]
        self.events: list[list] = []  # Format: [[[task_id, agent_id, "action"], ...], ...] - nested by agent
        
    def start_timing(self):
        """Start timing the planning process."""
        self.start_time = time.time()
        self.planner_times = []
    
    def record_step_time(self):
        """Record the time for a planning step."""
        if self.start_time is not None:
            current_time = time.time()
            step_time = current_time - self.start_time
            self.planner_times.append(step_time)
            self.start_time = current_time
    
    def add_task(self, task_id: int, y: int, x: int):
        """Add a task to the logger.
        
        Args:
            task_id: Unique identifier for the task
            y: Y coordinate of the task location
            x: X coordinate of the task location
        """
        task = [task_id, y, x]
        if task not in self.tasks:
            self.tasks.append(task)
    
    def add_event(self, task_id: int, agent_id: int, action: str):
        """Add an event to the logger.
        
        Args:
            task_id: ID of the task related to this event
            agent_id: ID of the agent performing the action  
            action: Type of action ("assigned", "finished", etc.)
        """
        event = [task_id, agent_id, action]
        self.events.append(event)
    
    def set_events_by_agent(self, events_by_agent: dict):
        """Set events organized by agent.
        
        Args:
            events_by_agent: Dictionary mapping agent_id to list of events
        """
        self.events = []
        # Add events for each agent in order
        for agent_id in sorted(events_by_agent.keys()):
            agent_events = [[task_id, timestep, action] for task_id, agent_id, action, timestep in events_by_agent[agent_id]]
            self.events.append(agent_events)
    
    def clear_tasks_and_events(self):
        """Clear all tasks and events."""
        self.tasks = []
        self.events = []
    
    def format_start_positions(self, starts: Union[Config, OrientedConfig]) -> list[list]:
        """Format start positions for JSON output.
        
        Args:
            starts: Starting positions (either Config or OrientedConfig)
            
        Returns:
            List of [y, x, orientation] for each agent
        """
        formatted_starts = []
        
        for start in starts:
            if len(start) == 2:  # Config format (y, x)
                y, x = start
                # Default orientation mapping: 0=N, 1=E, 2=S, 3=W
                orientation = "N"  # Default to North for regular PIBT
            else:  # OrientedConfig format (y, x, orientation)
                y, x, orient_int = start
                # Map orientation integer to letter
                orientation_map = {
                    Orientation.NORTH: "N",
                    Orientation.EAST: "E", 
                    Orientation.SOUTH: "S",
                    Orientation.WEST: "W"
                }
                orientation = orientation_map.get(orient_int, "N")
            
            formatted_starts.append([y, x, orientation])
        
        return formatted_starts
    
    def configs_to_action_paths(self, configs: Union[Configs, OrientedConfigs]) -> list[str]:
        """Convert a sequence of configurations to action path strings.
        
        Args:
            configs: Sequence of configurations
            
        Returns:
            List of action strings for each agent (e.g., ["F,F,R,W", "W,F,F,F"])
        """
        if not configs or len(configs) < 2:
            return []
        
        num_agents = len(configs[0])
        agent_paths = [[] for _ in range(num_agents)]
        
        # Process each timestep
        for t in range(len(configs) - 1):
            current_config = configs[t]
            next_config = configs[t + 1]
            
            # Determine action for each agent
            for agent_id in range(num_agents):
                current_pos = current_config[agent_id]
                next_pos = next_config[agent_id]
                
                action = self._determine_action(current_pos, next_pos)
                agent_paths[agent_id].append(action)
        
        # Convert action lists to comma-separated strings (no spaces)
        return [",".join(path) for path in agent_paths]
    
    def _determine_action(self, current_pos: Union[tuple, list], next_pos: Union[tuple, list]) -> str:
        """Determine the action taken between two positions.
        
        Args:
            current_pos: Current position (y, x) or (y, x, orientation)
            next_pos: Next position (y, x) or (y, x, orientation)
            
        Returns:
            Action string ("F", "R", "C", "W")
        """
        # Handle both Config and OrientedConfig formats
        if len(current_pos) == 2:  # Config format (no orientation info)
            curr_y, curr_x = current_pos
            next_y, next_x = next_pos
            
            # For non-oriented configs, we can only detect movement vs no movement
            if (curr_y, curr_x) != (next_y, next_x):
                return "F"  # Some movement occurred
            else:
                return "W"  # No movement
                
        else:  # OrientedConfig format
            curr_y, curr_x, curr_orient = current_pos
            next_y, next_x, next_orient = next_pos
            
            # Check if position changed
            position_changed = (curr_y, curr_x) != (next_y, next_x)
            orientation_changed = curr_orient != next_orient
            
            if position_changed and orientation_changed:
                # Both position and orientation changed - this could be a complex action
                # For simplicity, prioritize the movement
                return "F"
            elif position_changed and not orientation_changed:
                # Position changed but orientation stayed same - forward movement
                return "F"
            elif not position_changed and orientation_changed:
                # Only orientation changed - rotation
                rotation_diff = (next_orient - curr_orient) % 4
                if rotation_diff == 1:
                    return "R"  # Clockwise
                elif rotation_diff == 3:
                    return "C"  # Counterclockwise
                elif rotation_diff == 2:
                    return "R"  # 180-degree turn, choose clockwise for simplicity
            else:
                # Neither position nor orientation changed
                return "W"  # Wait
        
        return "W"  # Default fallback
    
    def create_log_entry(self, 
                        starts: Union[Config, OrientedConfig],
                        configs: Union[Configs, OrientedConfigs],
                        all_valid: bool = True) -> dict:
        """Create a complete log entry for the planning result.
        
        Args:
            starts: Starting positions
            configs: Sequence of configurations (the planned path)
            all_valid: Whether all agents reached their goals
            
        Returns:
            Dictionary containing the log entry in the required format
        """
        # Calculate team size
        team_size = len(starts)
        
        # Format start positions
        start_positions = self.format_start_positions(starts)
        
        # Convert configs to action paths
        actual_paths = self.configs_to_action_paths(configs)
        
        # Create the log entry
        log_entry = {
            "actionModel": self.actionModel,
            "AllValid": "Yes" if all_valid else "No",
            "teamSize": team_size,
            "start": start_positions,
            "actualPaths": actual_paths,
            "plannerTimes": self.planner_times.copy(),
            "errors": [],  # Left empty as requested
            "events": self.events.copy(),  # Now populated with actual events
            "tasks": self.tasks.copy()     # Now populated with actual tasks
        }
        
        return log_entry
    
    def save_to_file(self, 
                    log_entry: dict, 
                    filepath: Union[str, Path],
                    indent: int = 2):
        """Save log entry to a JSON file.
        
        Args:
            log_entry: The log entry dictionary
            filepath: Path to save the JSON file
            indent: JSON indentation for pretty printing
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=indent)
    
    def log_planning_result(self,
                           starts: Union[Config, OrientedConfig],
                           configs: Union[Configs, OrientedConfigs],
                           goals: Union[Config, OrientedConfig],
                           filepath: Optional[Union[str, Path]] = None,
                           all_valid: Optional[bool] = None) -> dict:
        """Complete logging workflow: create entry and optionally save to file.
        
        Args:
            starts: Starting positions
            configs: Sequence of configurations (the planned path)
            goals: Goal positions
            filepath: Optional path to save JSON file
            all_valid: Whether all agents reached goals (auto-detected if None)
            
        Returns:
            The log entry dictionary
        """
        # Auto-detect if all agents reached goals
        if all_valid is None:
            if configs:
                final_config = configs[-1]
                all_valid = self._check_all_goals_reached(final_config, goals)
            else:
                all_valid = False
        
        # Create log entry
        log_entry = self.create_log_entry(starts, configs, all_valid)
        
        # Save to file if requested
        if filepath is not None:
            self.save_to_file(log_entry, filepath)
        
        return log_entry
    
    def _check_all_goals_reached(self, 
                                final_config: Union[Config, OrientedConfig], 
                                goals: Union[Config, OrientedConfig]) -> bool:
        """Check if all agents reached their goals.
        
        Args:
            final_config: Final configuration
            goals: Goal positions
            
        Returns:
            True if all agents reached their goals
        """
        for i, (final_pos, goal_pos) in enumerate(zip(final_config, goals)):
            # Extract position coordinates (ignore orientation for goal check)
            if len(final_pos) == 2:
                final_y, final_x = final_pos
            else:
                final_y, final_x = final_pos[0], final_pos[1]
            
            if len(goal_pos) == 2:
                goal_y, goal_x = goal_pos
            else:
                goal_y, goal_x = goal_pos[0], goal_pos[1]
            
            if (final_y, final_x) != (goal_y, goal_x):
                return False
        
        return True
