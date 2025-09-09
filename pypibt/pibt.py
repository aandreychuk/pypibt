import numpy as np
from typing import Optional
from pathlib import Path

from .dist_table import OrientedDistTable
from .enums import (
    Action, Orientation, Config, Grid, 
    OrientedConfig, OrientedConfigs, OrientedCoord
)
from .mapf_utils import get_multi_action_operations, apply_action_sequence
from .action_sequences import generate_unique_action_sequences
from .task_manager import TaskManager
from .logger import PIBTLogger
from .solution_verifier import SolutionVerifier


class MultiActionPIBT:
    """PIBT variant that uses multi-action operations to better handle collision resolution."""
    
    def __init__(self, grid: Grid, starts: Config, task_manager: TaskManager, 
                 operation_length: int = 3, seed: int = 0, enable_logging: bool = False,
                 enable_verification: bool = True):
        self.grid = grid
        # Convert starts to oriented coordinates (all agents start with orientation 0 - North)
        self.starts: OrientedConfig = [(y, x, Orientation.NORTH) for y, x in starts]
        self.N = len(self.starts)
        self.operation_length = operation_length
        
        # Task manager is mandatory for dynamic goal assignment
        self.task_manager = task_manager
        self.goals = self.task_manager.get_current_goals()

        # Pre-generate only unique, meaningful action sequences for this operation length
        self.action_sequences = generate_unique_action_sequences(operation_length)

        # distance table with orientation support
        self.dist_tables = [OrientedDistTable(grid, goal) for goal in self.goals]

        # cache - now needs to handle 3D coordinates (y, x, orientation)
        self.NIL = self.N  # meaning \bot
        self.NIL_ORIENTED_COORD: OrientedCoord = (*self.grid.shape, -1)  # meaning \bot

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)
        
        # Initialize logger if enabled
        self.logger: Optional[PIBTLogger] = None
        if enable_logging:
            self.logger = PIBTLogger(actionModel="MAPF_T")
        
        # Initialize solution verifier if enabled
        self.verifier: Optional[SolutionVerifier] = None
        self.enable_verification = enable_verification
        if enable_verification:
            self.verifier = SolutionVerifier(grid)
    
    def _update_distance_tables(self):
        """Update distance tables when goals change."""
        self.dist_tables = [OrientedDistTable(self.grid, goal) for goal in self.goals]

    def is_operation_collision_free(self, agent_id: int, operation_states: list[OrientedCoord], 
                                   Q_from: OrientedConfig, reserved_operations: dict[int, list[OrientedCoord]], 
                                   visited_agents: set[int]) -> bool:
        """Check if an operation is collision-free using direct state comparison."""
        
        # Check against all already planned agents
        for other_agent_id, other_operation in reserved_operations.items():
            if other_agent_id == agent_id:
                continue
                
            # Check vertex collisions - same position at same time
            min_length = min(len(operation_states), len(other_operation))
            for t in range(min_length):
                my_pos = (operation_states[t][0], operation_states[t][1])
                other_pos = (other_operation[t][0], other_operation[t][1])
                
                if my_pos == other_pos:
                    return False  # Vertex collision
            
            # Check edge collisions - swapping positions
            for t in range(1, min_length):
                my_prev = (operation_states[t-1][0], operation_states[t-1][1])
                my_curr = (operation_states[t][0], operation_states[t][1])
                other_prev = (other_operation[t-1][0], other_operation[t-1][1])
                other_curr = (other_operation[t][0], other_operation[t][1])
                
                # Check if agents are swapping positions
                if my_prev == other_curr and my_curr == other_prev and my_prev != my_curr:
                    return False  # Edge collision
        
        # Check against unplanned agents (they stay at current position)
        for other_agent_id in range(self.N):
            if (other_agent_id == agent_id or 
                other_agent_id in reserved_operations or 
                other_agent_id in visited_agents):
                continue
                
            other_pos = (Q_from[other_agent_id][0], Q_from[other_agent_id][1])
            
            # Check if we collide with this stationary agent at any time
            for t, state in enumerate(operation_states):
                my_pos = (state[0], state[1])
                if my_pos == other_pos:
                    return False  # Collision with stationary agent
        
        return True

    def _find_collision_agents(self, agent_id: int, operation_states: list[OrientedCoord], 
                              Q_from: OrientedConfig, reserved_operations: dict[int, list[OrientedCoord]], 
                              visited_agents: set[int]) -> set[int]:
        """Find unplanned agents that would collide with the proposed operation."""
        collision_agents = set()
        
        for t, state in enumerate(operation_states):
            if t == 0:  # Skip current position at t=0
                continue
                
            pos = (state[0], state[1])
            
            # Check against unplanned agents (those that haven't been processed yet)
            for j in range(self.N):
                if j == agent_id or j in reserved_operations or j in visited_agents:
                    continue
                
                # Check if unplanned agent j would be at this position at time t
                current_pos_j = (Q_from[j][0], Q_from[j][1])
                
                # Unplanned agents are assumed to stay at their current position
                # So they occupy their current position for all future timesteps
                if pos == current_pos_j:
                    collision_agents.add(j)
        
        return collision_agents

    def funcMultiActionPIBT(self, Q_from: OrientedConfig, reserved_operations: dict[int, list[OrientedCoord]], 
                           visited_agents: set[int], i: int) -> bool:
        """Simplified multi-action PIBT function with direct collision checking."""
        
        if i in visited_agents:
            return True  # Already processed this agent in this round
            
        # Get all valid multi-action operations from current state using pre-computed sequences
        operations = get_multi_action_operations(self.grid, Q_from[i], self.action_sequences)
        
        # Shuffle for tie-breaking
        self.rng.shuffle(operations)
        
        # Sort by distance to goal (final state of operation)
        operations = sorted(operations, key=lambda op: self.dist_tables[i].get(op[1]))

        # Try each operation
        for action_sequence, final_state in operations:
            # Get all states during this operation
            operation_states = apply_action_sequence(Q_from[i], action_sequence)
            
            # Direct collision check - much simpler and more reliable
            if not self.is_operation_collision_free(i, operation_states, Q_from, reserved_operations, visited_agents):
                continue
            
            # Find unplanned agents that would collide with this operation
            collision_agents = self._find_collision_agents(i, operation_states, Q_from, reserved_operations, visited_agents)
            
            # SIMPLE RULE: Skip operations that collide with MORE THAN 1 agent
            if len(collision_agents) > 1:
                continue
            
            # Try to resolve collision with the single agent (if any)
            can_resolve = True
            if len(collision_agents) == 1:
                j = list(collision_agents)[0]
                temp_visited = visited_agents.copy()
                temp_visited.add(i)
                
                # Prevent infinite recursion and cycles
                if j in temp_visited or len(temp_visited) > self.N:
                    can_resolve = False
                else:
                    # Recursively plan for agent j to move it out of the way
                    if not self.funcMultiActionPIBT(Q_from, reserved_operations, temp_visited, j):
                        can_resolve = False
                    else:
                        visited_agents.add(j)
            
            if can_resolve:
                # Reserve this operation for agent i
                reserved_operations[i] = operation_states
                visited_agents.add(i)
                return True
        
        # Could not find a valid operation
        # Reserve stay operation as fallback
        stay_states = apply_action_sequence(Q_from[i], [Action.WAIT] * self.operation_length)
        reserved_operations[i] = stay_states
        visited_agents.add(i)
        return False

    def step(self, Q_from: OrientedConfig, priorities: list[float]) -> OrientedConfig:
        """Execute one multi-action step with simplified collision detection."""
        # Reserved operations for each agent
        reserved_operations: dict[int, list[OrientedCoord]] = {}
        visited_agents: set[int] = set()
        
        # Sort agents by priority
        A = sorted(list(range(self.N)), key=lambda i: priorities[i], reverse=True)
        
        # Plan operations for each agent
        for i in A:
            if i not in visited_agents:
                # Attempt to plan for agent i using direct collision checking
                success = self.funcMultiActionPIBT(Q_from, reserved_operations, visited_agents, i)
        
        # Execute first step of each agent's operation
        Q_to = []
        for i in range(self.N):
            if i in reserved_operations and len(reserved_operations[i]) > 1:
                Q_to.append(reserved_operations[i][1])  # First step of operation
            else:
                Q_to.append(Q_from[i])  # Stay in place
        
        # Final validation - check for any collisions in the result
        positions = {}
        for i, pos in enumerate(Q_to):
            pos_2d = (pos[0], pos[1])
            if pos_2d in positions:
                print(f"🚨 CRITICAL ERROR: Agents {i} and {positions[pos_2d]} both at position {pos_2d}")
                print(f"   Agent {i} operation: {reserved_operations.get(i, 'NONE')}")
                print(f"   Agent {positions[pos_2d]} operation: {reserved_operations.get(positions[pos_2d], 'NONE')}")
            else:
                positions[pos_2d] = i
        
        return Q_to

    def run(self, max_timestep: int = 1000, log_filepath: Optional[str] = None) -> OrientedConfigs:
        """Run the multi-action PIBT algorithm with task management and optional logging."""
        # Initialize logging if enabled
        if self.logger:
            # Log initial task assignments
            self.task_manager.log_initial_assignments(self.logger)
            
            # Start timing
            self.logger.start_timing()
        
        # define priorities (based on oriented distance)
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / (self.grid.size * 4))
        print(self.starts)
        
        # main loop, generate sequence of configurations
        configs = [self.starts]
        timestep = 0
        
        # Verify initial configuration if verification is enabled
        if self.enable_verification and self.verifier:
            is_valid, errors = self.verifier.verify_step(None, self.starts, 0, verbose=False)
            if not is_valid:
                print(f"❌ Initial configuration is invalid! Found {len(errors)} errors:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  - {error.error_type}: {error.message}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more errors")
        
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)
            timestep += 1

            # Verify the step if verification is enabled
            if self.enable_verification and self.verifier:
                is_valid, errors = self.verifier.verify_step(configs[-2], Q, timestep, verbose=False)
                if not is_valid:
                    print(f"❌ Step {timestep} is invalid! Found {len(errors)} errors:")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"  - {error.error_type}: {error.message}")
                    if len(errors) > 3:
                        print(f"  ... and {len(errors) - 3} more errors")
                    # Continue execution but warn user
                    print("⚠️  Continuing execution despite validation errors...")

            # Record step timing if logging is enabled
            if self.logger:
                self.logger.record_step_time()

            # Convert oriented positions to regular positions for task checking
            current_positions = [(pos[0], pos[1]) for pos in Q]
            print(timestep, Q, "\n")#self.goals)
            # Check for completed tasks and get new assignments
            completed_agents = self.task_manager.check_and_update_completed_tasks(current_positions, timestep, self.logger)
            
            # Update goals if any tasks were completed
            if completed_agents:
                self.goals = self.task_manager.get_current_goals()
                self._update_distance_tables()
            
            # Check if we should continue (task manager controls termination)
            if not self.task_manager.has_active_tasks():
                break
            
            # Update priorities based on current goals (which may have changed)
            for i in range(self.N):
                pos_q = (Q[i][0], Q[i][1])  # current position
                if pos_q != self.goals[i]:
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])

        # Final verification of the complete solution (LMAPF mode - no goal checking)
        if self.enable_verification and self.verifier:
            print("\n🔍 Performing final solution verification (LMAPF mode)...")
            # For LMAPF, we only verify collision-free movement, not goal achievement
            is_valid, errors = self.verifier.verify_solution(
                configs, starts=self.starts, goals=None, verbose=True
            )
            
            if not is_valid:
                # Filter out goal-related errors since they're not relevant for LMAPF
                collision_errors = [e for e in errors if e.error_type not in ['goal_mismatch', 'goal_count_mismatch']]
                if collision_errors:
                    error_summary = self.verifier.get_error_summary(collision_errors)
                    print(f"\n📊 Collision Error Summary:")
                    for error_type, count in error_summary.items():
                        print(f"  - {error_type}: {count} occurrences")
                else:
                    print("✅ No collision errors found! (Goal mismatches ignored in LMAPF mode)")

        # Handle logging at the end
        if self.logger and log_filepath:
            # Convert goals to regular config format for logging
            final_goals = [(y, x) for y, x in self.goals]
            self.logger.log_planning_result(
                starts=[(y, x) for y, x, _ in self.starts],  # Convert to regular config
                configs=configs,
                goals=final_goals,
                filepath=log_filepath
            )
            print(f"Logging results saved to: {log_filepath}")
            print(f"Logged {len(self.logger.tasks)} actually used tasks (out of {len(self.task_manager.task_pool)} total tasks)")

        return configs
