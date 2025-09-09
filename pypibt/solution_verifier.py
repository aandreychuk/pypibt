"""
Solution verifier for MAPF problems with detailed validation and error reporting.
"""

from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

from .enums import Grid, Config, OrientedConfig, OrientedConfigs
from .mapf_utils import is_valid_coord, get_neighbors


@dataclass
class ValidationError:
    """Represents a validation error with detailed information."""
    error_type: str
    timestep: int
    agent_id: Optional[int] = None
    other_agent_id: Optional[int] = None
    position: Optional[Tuple[int, int]] = None
    message: str = ""


class SolutionVerifier:
    """
    Comprehensive solution verifier for MAPF problems.
    
    Validates:
    - Agents stay within grid bounds
    - Agents don't move to obstacles  
    - No vertex collisions (same cell occupation)
    - No edge collisions (agents swapping positions)
    - Movement continuity (agents can only move to adjacent cells or stay)
    """
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.height, self.width = grid.shape
    
    def verify_solution(self, solution: OrientedConfigs, 
                       starts: Optional[OrientedConfig] = None,
                       goals: Optional[Config] = None,
                       verbose: bool = True) -> Tuple[bool, List[ValidationError]]:
        """
        Verify a complete solution path.
        
        Args:
            solution: List of configurations (timesteps)
            starts: Optional starting configuration to verify against
            goals: Optional goal configuration to verify against  
            verbose: Whether to print validation results
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not solution:
            errors.append(ValidationError("empty_solution", -1, message="Solution is empty"))
            return False, errors
            
        num_agents = len(solution[0])
        
        # Verify starts match if provided
        if starts is not None:
            if len(starts) != num_agents:
                errors.append(ValidationError("start_count_mismatch", 0, 
                                            message=f"Start count {len(starts)} != agent count {num_agents}"))
            else:
                for i, (start_pos, actual_pos) in enumerate(zip(starts, solution[0])):
                    if start_pos != actual_pos:
                        errors.append(ValidationError("start_mismatch", 0, i, 
                                                    position=(actual_pos[0], actual_pos[1]),
                                                    message=f"Agent {i} start mismatch: expected {start_pos}, got {actual_pos}"))
        
        # Verify goals match if provided
        if goals is not None:
            if len(goals) != num_agents:
                errors.append(ValidationError("goal_count_mismatch", len(solution)-1,
                                            message=f"Goal count {len(goals)} != agent count {num_agents}"))
            else:
                final_positions = [(pos[0], pos[1]) for pos in solution[-1]]
                for i, (goal_pos, actual_pos) in enumerate(zip(goals, final_positions)):
                    if goal_pos != actual_pos:
                        errors.append(ValidationError("goal_mismatch", len(solution)-1, i,
                                                    position=actual_pos,
                                                    message=f"Agent {i} goal mismatch: expected {goal_pos}, got {actual_pos}"))
        
        # Verify each timestep
        for t in range(len(solution)):
            config = solution[t]
            
            # Check agent count consistency
            if len(config) != num_agents:
                errors.append(ValidationError("agent_count_inconsistent", t,
                                            message=f"Timestep {t} has {len(config)} agents, expected {num_agents}"))
                continue
                
            # Verify each agent's position
            for i, pos in enumerate(config):
                agent_errors = self._verify_agent_position(pos, t, i)
                errors.extend(agent_errors)
            
            # Verify movement continuity (except first timestep)
            if t > 0:
                prev_config = solution[t-1]
                for i in range(num_agents):
                    movement_errors = self._verify_movement_continuity(
                        prev_config[i], config[i], t, i)
                    errors.extend(movement_errors)
            
            # Verify no collisions at this timestep
            collision_errors = self._verify_no_collisions(config, t)
            errors.extend(collision_errors)
            
            # Verify no edge collisions (if not first timestep)
            if t > 0:
                edge_errors = self._verify_no_edge_collisions(solution[t-1], config, t)
                errors.extend(edge_errors)
        
        is_valid = len(errors) == 0
        
        if verbose:
            if is_valid:
                print(f"✅ Solution is VALID! Verified {len(solution)} timesteps with {num_agents} agents.")
            else:
                print(f"❌ Solution is INVALID! Found {len(errors)} errors:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  - {error.error_type} at t={error.timestep}: {error.message}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
        
        return is_valid, errors
    
    def _verify_agent_position(self, pos: Tuple[int, int, int], timestep: int, 
                              agent_id: int) -> List[ValidationError]:
        """Verify a single agent's position is valid."""
        errors = []
        y, x, orientation = pos
        
        # Check bounds
        if y < 0 or y >= self.height:
            errors.append(ValidationError("out_of_bounds_y", timestep, agent_id,
                                        position=(y, x),
                                        message=f"Agent {agent_id} y={y} out of bounds [0, {self.height-1}]"))
        
        if x < 0 or x >= self.width:
            errors.append(ValidationError("out_of_bounds_x", timestep, agent_id,
                                        position=(y, x), 
                                        message=f"Agent {agent_id} x={x} out of bounds [0, {self.width-1}]"))
        
        # Check orientation bounds
        if orientation < 0 or orientation > 3:
            errors.append(ValidationError("invalid_orientation", timestep, agent_id,
                                        position=(y, x),
                                        message=f"Agent {agent_id} orientation={orientation} invalid [0-3]"))
        
        # Check obstacle collision (only if position is in bounds)
        if (0 <= y < self.height and 0 <= x < self.width and 
            not self.grid[y, x]):
            errors.append(ValidationError("obstacle_collision", timestep, agent_id,
                                        position=(y, x),
                                        message=f"Agent {agent_id} at obstacle position ({y}, {x})"))
        
        return errors
    
    def _verify_movement_continuity(self, prev_pos: Tuple[int, int, int], 
                                   curr_pos: Tuple[int, int, int],
                                   timestep: int, agent_id: int) -> List[ValidationError]:
        """Verify agent movement is continuous (adjacent cells or same cell)."""
        errors = []
        
        prev_y, prev_x, prev_orient = prev_pos
        curr_y, curr_x, curr_orient = curr_pos
        
        # Position change
        dy = curr_y - prev_y
        dx = curr_x - prev_x
        
        # Valid moves: stay in place or move to adjacent cell
        valid_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        
        if (dy, dx) not in valid_moves:
            errors.append(ValidationError("invalid_movement", timestep, agent_id,
                                        position=(curr_y, curr_x),
                                        message=f"Agent {agent_id} invalid move from ({prev_y}, {prev_x}) to ({curr_y}, {curr_x})"))
        
        # Check that if position changed, the previous position was valid and connected
        if (dy, dx) != (0, 0):
            if (0 <= prev_y < self.height and 0 <= prev_x < self.width and
                0 <= curr_y < self.height and 0 <= curr_x < self.width):
                
                prev_neighbors = get_neighbors(self.grid, (prev_y, prev_x))
                if (curr_y, curr_x) not in prev_neighbors:
                    errors.append(ValidationError("disconnected_movement", timestep, agent_id,
                                                position=(curr_y, curr_x),
                                                message=f"Agent {agent_id} moved to disconnected cell ({curr_y}, {curr_x}) from ({prev_y}, {prev_x})"))
        
        return errors
    
    def _verify_no_collisions(self, config: List[Tuple[int, int, int]], 
                             timestep: int) -> List[ValidationError]:
        """Verify no two agents occupy the same position."""
        errors = []
        positions = {}
        
        for i, pos in enumerate(config):
            pos_2d = (pos[0], pos[1])  # Only consider x, y for collision
            
            if pos_2d in positions:
                other_agent = positions[pos_2d]
                errors.append(ValidationError("vertex_collision", timestep, i, other_agent,
                                            position=pos_2d,
                                            message=f"Agents {i} and {other_agent} both at position {pos_2d}"))
            else:
                positions[pos_2d] = i
        
        return errors
    
    def _verify_no_edge_collisions(self, prev_config: List[Tuple[int, int, int]], 
                                  curr_config: List[Tuple[int, int, int]],
                                  timestep: int) -> List[ValidationError]:
        """Verify no two agents swap positions (edge collision)."""
        errors = []
        
        for i in range(len(curr_config)):
            prev_pos_i = (prev_config[i][0], prev_config[i][1])
            curr_pos_i = (curr_config[i][0], curr_config[i][1])
            
            for j in range(i + 1, len(curr_config)):
                prev_pos_j = (prev_config[j][0], prev_config[j][1])
                curr_pos_j = (curr_config[j][0], curr_config[j][1])
                
                # Check if agents swapped positions
                if (curr_pos_i == prev_pos_j and curr_pos_j == prev_pos_i and 
                    curr_pos_i != curr_pos_j):  # Make sure they actually moved
                    errors.append(ValidationError("edge_collision", timestep, i, j,
                                                position=curr_pos_i,
                                                message=f"Agents {i} and {j} swapped positions: {prev_pos_i} ↔ {prev_pos_j}"))
        
        return errors
    
    def verify_step(self, prev_config: Optional[List[Tuple[int, int, int]]], 
                   curr_config: List[Tuple[int, int, int]],
                   timestep: int, verbose: bool = False) -> Tuple[bool, List[ValidationError]]:
        """
        Verify a single step transition.
        
        Args:
            prev_config: Previous configuration (None for first step)
            curr_config: Current configuration
            timestep: Current timestep
            verbose: Whether to print results
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Verify current positions
        for i, pos in enumerate(curr_config):
            agent_errors = self._verify_agent_position(pos, timestep, i)
            errors.extend(agent_errors)
        
        # Verify movement and collisions if we have previous step
        if prev_config is not None:
            # Verify movement continuity
            for i in range(len(curr_config)):
                movement_errors = self._verify_movement_continuity(
                    prev_config[i], curr_config[i], timestep, i)
                errors.extend(movement_errors)
            
            # Verify no edge collisions
            edge_errors = self._verify_no_edge_collisions(prev_config, curr_config, timestep)
            errors.extend(edge_errors)
        
        # Verify no vertex collisions
        collision_errors = self._verify_no_collisions(curr_config, timestep)
        errors.extend(collision_errors)
        
        is_valid = len(errors) == 0
        
        if verbose and not is_valid:
            print(f"❌ Step {timestep} is INVALID! Found {len(errors)} errors:")
            for error in errors:
                print(f"  - {error.error_type}: {error.message}")
        
        return is_valid, errors
    
    def get_error_summary(self, errors: List[ValidationError]) -> dict:
        """Get a summary of error types and counts."""
        summary = {}
        for error in errors:
            error_type = error.error_type
            if error_type not in summary:
                summary[error_type] = 0
            summary[error_type] += 1
        return summary
