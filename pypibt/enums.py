"""
Shared enums and types for the pypibt package.
"""

from enum import IntEnum
from typing import TypeAlias

import numpy as np

# Orientation enum (0=North, 1=East, 2=South, 3=West)
class Orientation(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# Action enum
class Action(IntEnum):
    MOVE_FORWARD = 0
    ROTATE_CLOCKWISE = 1
    ROTATE_COUNTERCLOCKWISE = 2
    WAIT = 3

# Type aliases
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
OrientedCoord: TypeAlias = tuple[int, int, int]  # (y, x, orientation)
Config: TypeAlias = list[Coord]
OrientedConfig: TypeAlias = list[OrientedCoord]
Configs: TypeAlias = list[Config]
OrientedConfigs: TypeAlias = list[OrientedConfig]
