from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .enums import Action, Coord, Grid, OrientedCoord
from .mapf_utils import (
    get_neighbors, get_oriented_neighbors, 
    is_valid_coord, is_valid_oriented_coord, apply_action
)


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    Q: deque = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self):
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.grid.size, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size:
            return self.table[target]

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.grid.size
    
    def get_oriented(self, target: OrientedCoord) -> int:
        """Get distance from goal to oriented coordinate (position only)."""
        y, x, _ = target  # Ignore orientation for distance calculation
        return self.get((y, x))


@dataclass
class OrientedDistTable:
    """Distance table that accounts for orientation and action costs."""
    grid: Grid
    goal: Coord
    Q: deque = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix (height, width, 4 orientations)

    def __post_init__(self):
        # Create 3D table: (height, width, 4 orientations)
        height, width = self.grid.shape
        self.table = np.full((height, width, 4), self.grid.size * 4, dtype=int)
        
        # Initialize goal states - all orientations at goal have cost 0
        goal_y, goal_x = self.goal
        for orientation in range(4):
            self.table[goal_y, goal_x, orientation] = 0
        
        # Compute distances using full BFS from goal
        self._compute_all_distances()

    def _compute_all_distances(self):
        """Compute distances from goal to all oriented coordinates using BFS."""
        Q = deque()
        
        # Start BFS from all goal orientations
        goal_y, goal_x = self.goal
        for orientation in range(4):
            Q.append((goal_y, goal_x, orientation))
        
        # BFS to compute distances backwards from goal
        while Q:
            curr_y, curr_x, curr_orient = Q.popleft()
            curr_cost = self.table[curr_y, curr_x, curr_orient]
            
            # Try all possible predecessor states that could reach current state
            # For each action, find which states could perform that action to reach current state
            
            # 1. States that could move forward to reach current state
            reverse_orient = (curr_orient + 2) % 4  # Opposite direction
            if reverse_orient == 0:  # North - predecessor moved from south
                pred_y, pred_x = curr_y + 1, curr_x
            elif reverse_orient == 1:  # East - predecessor moved from west  
                pred_y, pred_x = curr_y, curr_x - 1
            elif reverse_orient == 2:  # South - predecessor moved from north
                pred_y, pred_x = curr_y - 1, curr_x
            elif reverse_orient == 3:  # West - predecessor moved from east
                pred_y, pred_x = curr_y, curr_x + 1
            
            pred_state = (pred_y, pred_x, reverse_orient)
            if is_valid_oriented_coord(self.grid, pred_state):
                new_cost = curr_cost + 1
                if new_cost < self.table[pred_y, pred_x, reverse_orient]:
                    self.table[pred_y, pred_x, reverse_orient] = new_cost
                    Q.append(pred_state)
            
            # 2. States that could rotate clockwise to reach current state
            pred_orient = (curr_orient - 1) % 4
            pred_state = (curr_y, curr_x, pred_orient)
            if is_valid_oriented_coord(self.grid, pred_state):
                new_cost = curr_cost + 1
                if new_cost < self.table[curr_y, curr_x, pred_orient]:
                    self.table[curr_y, curr_x, pred_orient] = new_cost
                    Q.append(pred_state)
            
            # 3. States that could rotate counterclockwise to reach current state
            pred_orient = (curr_orient + 1) % 4
            pred_state = (curr_y, curr_x, pred_orient)
            if is_valid_oriented_coord(self.grid, pred_state):
                new_cost = curr_cost + 1
                if new_cost < self.table[curr_y, curr_x, pred_orient]:
                    self.table[curr_y, curr_x, pred_orient] = new_cost
                    Q.append(pred_state)
            
            # 4. States that could wait to reach current state (same state)
            # This is just the current state itself, but we already processed it

    def get(self, target: OrientedCoord) -> int:
        """Get minimum cost from any goal orientation to target oriented coordinate."""
        if not is_valid_oriented_coord(self.grid, target):
            return self.grid.size * 4

        y, x, orientation = target
        return self.table[y, x, orientation]
