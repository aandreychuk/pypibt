import os
import re

import numpy as np

from .enums import (
    Action, 
    Orientation, 
    Grid, 
    Coord, 
    OrientedCoord, 
    Config, 
    OrientedConfig, 
    Configs, 
    OrientedConfigs
)
from .action_sequences import (
    generate_action_sequences,
    generate_unique_action_sequences,
    action_sequence_to_string
)


def get_grid(map_file: str) -> Grid:
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str, N: int | None = None) -> tuple[Config, Config]:
    with open(scen_file, "r") as f:
        starts, goals = [], []
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def get_agents(agents_file: str, N: int | None = None) -> Config:
    """Load agent starting positions from a CSV agents file.
    
    CSV format:
    agent_id,row,col
    0,5,10
    1,2,7
    ...
    
    Args:
        agents_file: Path to the CSV agents file
        N: Optional limit on number of agents to load
        
    Returns:
        List of starting positions [(y, x), ...] sorted by agent_id
    """
    import csv
    
    agents = []
    with open(agents_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                agent_id = int(row['agent_id'])
                y = int(row['row'])  # row = y coordinate
                x = int(row['col'])  # col = x coordinate
                agents.append((agent_id, y, x))
            except (ValueError, KeyError):
                continue  # Skip invalid rows
    
    # Sort by agent_id to ensure consistent ordering
    agents.sort(key=lambda x: x[0])
    
    # Extract positions and limit if requested
    starts = [(y, x) for agent_id, y, x in agents]
    if N is not None:
        starts = starts[:N]
    
    return starts


def get_tasks(tasks_file: str, grid_width: int) -> list[tuple[int, int, int]]:
    """Load tasks from a CSV tasks file.
    
    CSV format:
    targets
    15
    23
    47
    ...
    
    Each target is identified by cell_id = y*width + x
    
    Args:
        tasks_file: Path to the CSV tasks file
        grid_width: Width of the grid (needed to convert cell_id to y, x)
        
    Returns:
        List of tasks [(task_id, y, x), ...] where task_id is sequential
    """
    import csv
    
    tasks = []
    with open(tasks_file, "r") as f:
        reader = csv.DictReader(f)
        task_id = 0
        for row in reader:
            try:
                cell_id = int(row['targets'])
                # Convert cell_id to y, x coordinates
                y = cell_id // grid_width
                x = cell_id % grid_width
                tasks.append((task_id, y, x))
                task_id += 1
            except (ValueError, KeyError):
                continue  # Skip invalid rows
    
    return tasks


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def is_valid_oriented_coord(grid: Grid, oriented_coord: OrientedCoord) -> bool:
    y, x, orientation = oriented_coord
    if orientation < 0 or orientation > 3:
        return False
    return is_valid_coord(grid, (y, x))


def get_forward_coord(coord: Coord, orientation: int) -> Coord:
    """Get the coordinate in front of the agent given current position and orientation."""
    y, x = coord
    if orientation == Orientation.NORTH:
        return (y - 1, x)
    elif orientation == Orientation.EAST:
        return (y, x + 1)
    elif orientation == Orientation.SOUTH:
        return (y + 1, x)
    elif orientation == Orientation.WEST:
        return (y, x - 1)
    else:
        raise ValueError(f"Invalid orientation: {orientation}")


def apply_action(oriented_coord: OrientedCoord, action: Action) -> OrientedCoord:
    """Apply an action to an oriented coordinate and return the new oriented coordinate."""
    y, x, orientation = oriented_coord
    
    if action == Action.MOVE_FORWARD:
        new_y, new_x = get_forward_coord((y, x), orientation)
        return (new_y, new_x, orientation)
    elif action == Action.ROTATE_CLOCKWISE:
        new_orientation = (orientation + 1) % 4
        return (y, x, new_orientation)
    elif action == Action.ROTATE_COUNTERCLOCKWISE:
        new_orientation = (orientation - 1) % 4
        return (y, x, new_orientation)
    elif action == Action.WAIT:
        return oriented_coord
    else:
        raise ValueError(f"Invalid action: {action}")


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh


def get_oriented_neighbors(grid: Grid, oriented_coord: OrientedCoord) -> list[OrientedCoord]:
    """Get all possible next oriented coordinates for an agent."""
    if not is_valid_oriented_coord(grid, oriented_coord):
        return []
    
    neighbors = []
    
    # Try all possible actions
    for action in Action:
        next_coord = apply_action(oriented_coord, action)
        
        # For move forward, check if the destination is valid
        if action == Action.MOVE_FORWARD:
            y, x, orientation = next_coord
            if is_valid_coord(grid, (y, x)):
                neighbors.append(next_coord)
        else:
            # Rotation and wait actions are always valid if current position is valid
            neighbors.append(next_coord)
    
    return neighbors


def apply_action_sequence(oriented_coord: OrientedCoord, actions: list[Action]) -> list[OrientedCoord]:
    """Apply a sequence of actions and return all intermediate states."""
    states = [oriented_coord]
    current_state = oriented_coord
    
    for action in actions:
        current_state = apply_action(current_state, action)
        states.append(current_state)
    
    return states


def get_multi_action_operations(grid: Grid, oriented_coord: OrientedCoord, 
                               pre_computed_sequences: list[list[Action]] = None, 
                               length: int = 3) -> list[tuple[list[Action], OrientedCoord]]:
    """Get all valid multi-action operations from an oriented coordinate.
    
    Args:
        grid: The grid
        oriented_coord: Starting oriented coordinate
        pre_computed_sequences: Pre-generated action sequences (if None, generates them)
        length: Length of action sequences (only used if pre_computed_sequences is None)
        
    Returns:
        List of (action_sequence, final_state) tuples where the operation is valid.
    """
    if not is_valid_oriented_coord(grid, oriented_coord):
        return []
    
    # Use pre-computed sequences if provided, otherwise generate them
    if pre_computed_sequences is None:
        action_sequences = generate_action_sequences(length)
    else:
        action_sequences = pre_computed_sequences
    
    valid_operations = []
    
    for actions in action_sequences:
        # Apply the action sequence step by step and check validity
        current_state = oriented_coord
        valid = True
        
        for action in actions:
            next_state = apply_action(current_state, action)
            
            # Check if this step is valid
            if not is_valid_oriented_coord(grid, next_state):
                valid = False
                break
                
            current_state = next_state
        
        if valid:
            valid_operations.append((actions, current_state))
    
    return valid_operations


def oriented_config_to_config(oriented_config: OrientedConfig) -> Config:
    """Convert oriented configuration to regular configuration (position only)."""
    return [(y, x) for y, x, _ in oriented_config]


def save_configs_for_visualizer(configs: Configs, filename: str) -> None:
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def save_oriented_configs_for_visualizer(oriented_configs: OrientedConfigs, filename: str) -> None:
    """Save oriented configurations for visualization (positions only)."""
    configs = [oriented_config_to_config(config) for config in oriented_configs]
    save_configs_for_visualizer(configs, filename)


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def validate_oriented_mapf_solution(
    grid: Grid,
    starts: OrientedConfig,
    goals: Config,  # Goals don't need orientation
    solution: OrientedConfigs,
) -> None:
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals (position only)
    goal_positions = [(y, x) for y, x, _ in solution[-1]]
    assert all(
        [u == v for (u, v) in zip(goals, goal_positions)]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity - must be reachable by one of the four actions
            assert v_i_now in [v_i_pre] + get_oriented_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision (position only)
            pos_i_now = (v_i_now[0], v_i_now[1])
            pos_i_pre = (v_i_pre[0], v_i_pre[1])
            
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                
                pos_j_now = (v_j_now[0], v_j_now[1])
                pos_j_pre = (v_j_pre[0], v_j_pre[1])
                
                assert not (pos_i_now == pos_j_now), "invalid solution, vertex collision"
                assert not (
                    pos_i_now == pos_j_pre and pos_i_pre == pos_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False


def is_valid_oriented_mapf_solution(
    grid: Grid,
    starts: OrientedConfig,
    goals: Config,
    solution: OrientedConfigs,
) -> bool:
    try:
        validate_oriented_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False
