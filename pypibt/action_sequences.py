"""
Action sequence generation and optimization for multi-action PIBT operations.

This module handles the generation of unique, canonical action sequences that
produce distinct movement patterns while eliminating redundant operations.
"""

from itertools import product

from .enums import Action, Orientation, OrientedCoord

# Global cache for generated sequences
_action_sequence_cache: dict[str, list[list[Action]]] = {}


def generate_action_sequences(length: int) -> list[list[Action]]:
    """Generate all possible action sequences of given length."""
    if length in _action_sequence_cache:
        return _action_sequence_cache[length]
    
    actions = [Action.MOVE_FORWARD, Action.ROTATE_CLOCKWISE, Action.ROTATE_COUNTERCLOCKWISE, Action.WAIT]
    sequences = [list(seq) for seq in product(actions, repeat=length)]
    
    # Cache the result
    _action_sequence_cache[length] = sequences
    return sequences


def generate_unique_action_sequences(length: int) -> list[list[Action]]:
    """Generate action sequences that result in unique location sequences.
    
    Uses a two-phase approach:
    1. Find all operations with unique location sequences
    2. Convert each operation to canonical form (no meaningless rotations)
    
    For length 3, this should find the 17 canonical operations:
    "WWW RRF RWF CWF WWF RFW CFW WFW RFF CFF FWW WFF FRF FCF FWF FFW FFF"
    """
    # Check cache first  
    cache_key = f"unique_locations_{length}"
    if cache_key in _action_sequence_cache:
        return _action_sequence_cache[cache_key]
    
    # Phase 1: Find all operations with unique location sequences
    unique_location_sequences = find_unique_location_sequences(length)
    
    # Phase 2: Convert each operation to canonical form
    canonical_operations = []
    for location_seq, operations in unique_location_sequences.items():
        # Pick any operation and convert it to canonical form
        canonical_op = convert_to_canonical_form(operations[0], length)
        canonical_operations.append(canonical_op)
    
    # Sort by preference for easier inspection/debugging
    canonical_operations.sort(key=lambda actions: (
        -actions.count(Action.MOVE_FORWARD),  # More moves first
        actions.count(Action.WAIT),           # Fewer waits first
        action_sequence_to_string(actions)    # Lexicographic for consistency
    ))
    
    # Cache the result
    _action_sequence_cache[cache_key] = canonical_operations
    return canonical_operations


def find_unique_location_sequences(length: int) -> dict:
    """Phase 1: Find all operations that produce unique location sequences.
    
    Returns:
        dict: Maps location_sequence -> list of operations that produce it
    """
    # Generate all possible action sequences
    all_sequences = generate_action_sequences(length)
    
    # Track unique location sequences
    location_to_operations = {}  # Maps location_sequence -> list of operations
    
    # Test from a canonical starting position and orientation
    start_state = (0, 0, Orientation.NORTH)
    
    for actions in all_sequences:
        # Get the sequence of locations visited during this operation
        location_sequence = get_location_sequence(start_state, actions)
        location_key = tuple(location_sequence)
        
        if location_key not in location_to_operations:
            location_to_operations[location_key] = []
        location_to_operations[location_key].append(actions)
    
    return location_to_operations


def convert_to_canonical_form(operation: list[Action], length: int) -> list[Action]:
    """Phase 2: Convert an operation to canonical form.
    
    The canonical form:
    - Eliminates meaningless rotation sequences (e.g., RC = 0 rotation)
    - Replaces meaningless rotations with waits
    - Should not end with rotations when equivalent wait sequences exist
    """
    canonical = operation.copy()
    
    # First, eliminate meaningless rotation pairs (RC or CR canceling out)
    canonical = eliminate_meaningless_rotations(canonical)
    
    # If the operation ends with rotations that don't affect final position,
    # replace trailing rotations with waits
    i = len(canonical) - 1
    while i >= 0 and canonical[i] in [Action.ROTATE_CLOCKWISE, Action.ROTATE_COUNTERCLOCKWISE]:
        canonical[i] = Action.WAIT
        i -= 1
    
    return canonical


def eliminate_meaningless_rotations(actions: list[Action]) -> list[Action]:
    """Eliminate meaningless rotation sequences like RC (clockwise+counterclockwise)."""
    result = actions.copy()
    
    # Keep processing until no more meaningless pairs are found
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            current = result[i]
            next_action = result[i + 1]
            
            # Check for canceling rotation pairs
            if ((current == Action.ROTATE_CLOCKWISE and next_action == Action.ROTATE_COUNTERCLOCKWISE) or
                (current == Action.ROTATE_COUNTERCLOCKWISE and next_action == Action.ROTATE_CLOCKWISE)):
                # Replace both rotations with waits
                result[i] = Action.WAIT
                result[i + 1] = Action.WAIT
                changed = True
                i += 2  # Skip the pair we just processed
            else:
                i += 1
    
    return result


def get_location_sequence(start_state: OrientedCoord, actions: list[Action]) -> list[tuple[int, int]]:
    """Get the sequence of (y, x) locations visited during action execution."""
    locations = [(start_state[0], start_state[1])]  # Start location
    current_state = start_state
    
    for action in actions:
        current_state = apply_action(current_state, action)
        locations.append((current_state[0], current_state[1]))
    
    return locations


def apply_action(oriented_coord: OrientedCoord, action: Action) -> OrientedCoord:
    """Apply an action to an oriented coordinate and return the new state."""
    y, x, orientation = oriented_coord
    
    if action == Action.MOVE_FORWARD:
        # Move forward in current orientation
        if orientation == Orientation.NORTH:
            y -= 1
        elif orientation == Orientation.EAST:
            x += 1
        elif orientation == Orientation.SOUTH:
            y += 1
        elif orientation == Orientation.WEST:
            x -= 1
    elif action == Action.ROTATE_CLOCKWISE:
        orientation = (orientation + 1) % 4
    elif action == Action.ROTATE_COUNTERCLOCKWISE:
        orientation = (orientation - 1) % 4
    # WAIT doesn't change anything
    
    return (y, x, orientation)


def action_sequence_to_string(actions: list[Action]) -> str:
    """Convert action sequence to string using F/R/C/W encoding."""
    mapping = {
        Action.MOVE_FORWARD: 'F',
        Action.ROTATE_CLOCKWISE: 'R', 
        Action.ROTATE_COUNTERCLOCKWISE: 'C',
        Action.WAIT: 'W'
    }
    return ''.join(mapping[action] for action in actions)


def clear_cache():
    """Clear the action sequence cache. Useful for testing."""
    global _action_sequence_cache
    _action_sequence_cache.clear()