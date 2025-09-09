import numpy as np
from typing import Optional

from .dist_table import OrientedDistTable
from .enums import (
    Action, Orientation, Config, Grid, 
    OrientedConfig, OrientedConfigs, OrientedCoord
)
from .mapf_utils import get_multi_action_operations, apply_action_sequence
from .action_sequences import generate_unique_action_sequences
from .task_manager import TaskManager


class MultiActionPIBT:
    """PIBT variant that uses multi-action operations to better handle collision resolution."""
    
    def __init__(self, grid: Grid, starts: Config, task_manager: TaskManager, 
                 operation_length: int = 3, seed: int = 0):
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
    
    def _update_distance_tables(self):
        """Update distance tables when goals change."""
        self.dist_tables = [OrientedDistTable(self.grid, goal) for goal in self.goals]

    def check_operation_collision(self, agent_id: int, operation_states: list[OrientedCoord], 
                                  reserved_operations: dict[int, list[OrientedCoord]]) -> bool:
        """Check if an operation collides with already reserved operations of other agents."""
        # Check collision with each timestep of the operation
        for t, state in enumerate(operation_states):
            pos = (state[0], state[1])  # position only
            
            # Check if this position is occupied by another agent at this timestep
            for other_agent, other_states in reserved_operations.items():
                if other_agent == agent_id:
                    continue
                    
                if t < len(other_states):
                    other_pos = (other_states[t][0], other_states[t][1])
                    if pos == other_pos:
                        return True  # Collision detected
                        
            # Also check edge collisions (agents swapping positions)
            if t > 0:
                prev_pos = (operation_states[t-1][0], operation_states[t-1][1])
                for other_agent, other_states in reserved_operations.items():
                    if other_agent == agent_id:
                        continue
                    if t < len(other_states) and t-1 >= 0:
                        other_pos = (other_states[t][0], other_states[t][1])
                        other_prev_pos = (other_states[t-1][0], other_states[t-1][1])
                        if pos == other_prev_pos and prev_pos == other_pos:
                            return True  # Edge collision detected
        
        return False

    def funcMultiActionPIBT(self, Q_from: OrientedConfig, reserved_operations: dict[int, list[OrientedCoord]], 
                           visited_agents: set[int], i: int) -> bool:
        """Multi-action PIBT function that can visit agents multiple times."""
        
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
            
            # Check for collisions with already reserved operations
            if self.check_operation_collision(i, operation_states, reserved_operations):
                continue
            
            # Check for collisions with agents that might need to be displaced
            collision_agents = set()
            for t, state in enumerate(operation_states):
                pos = (state[0], state[1])
                
                # Find agents currently at this position
                for j in range(self.N):
                    if j == i or j in reserved_operations:
                        continue
                    
                    current_pos = (Q_from[j][0], Q_from[j][1])
                    if pos == current_pos:
                        collision_agents.add(j)
            
            # Try to resolve collisions by recursively planning for conflicting agents
            can_resolve = True
            temp_visited = visited_agents.copy()
            temp_visited.add(i)
            
            for j in collision_agents:
                if not self.funcMultiActionPIBT(Q_from, reserved_operations, temp_visited, j):
                    can_resolve = False
                    break
            
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
        """Execute one multi-action step."""
        # Reserved operations for each agent
        reserved_operations: dict[int, list[OrientedCoord]] = {}
        visited_agents: set[int] = set()
        
        # Sort agents by priority
        A = sorted(list(range(self.N)), key=lambda i: priorities[i], reverse=True)
        
        # Plan operations for each agent
        for i in A:
            if i not in visited_agents:
                self.funcMultiActionPIBT(Q_from, reserved_operations, visited_agents, i)
        
        # Execute first step of each agent's operation
        Q_to = []
        for i in range(self.N):
            if i in reserved_operations and len(reserved_operations[i]) > 1:
                Q_to.append(reserved_operations[i][1])  # First step of operation
            else:
                Q_to.append(Q_from[i])  # Stay in place
        
        return Q_to

    def run(self, max_timestep: int = 1000) -> OrientedConfigs:
        """Run the multi-action PIBT algorithm with task management."""
        # define priorities (based on oriented distance)
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / (self.grid.size * 4))
        print(self.starts)
        # main loop, generate sequence of configurations
        configs = [self.starts]
        timestep = 0
        
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)
            timestep += 1

            # Convert oriented positions to regular positions for task checking
            current_positions = [(pos[0], pos[1]) for pos in Q]
            print(timestep, Q, "\n")#self.goals)
            # Check for completed tasks and get new assignments
            completed_agents = self.task_manager.check_and_update_completed_tasks(current_positions, timestep)
            
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

        return configs
