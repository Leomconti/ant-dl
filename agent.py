from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import random

@dataclass
class FoodSource:
    x: int
    y: int
    amount: float = 100.0

    def consume(self, amount: float = 1.0) -> float:
        """Consume food and return amount actually consumed"""
        if self.amount >= amount:
            self.amount -= amount
            return amount
        consumed = self.amount
        self.amount = 0
        return consumed

    @property
    def is_depleted(self) -> bool:
        return self.amount <= 0

class AntState(Enum):
    EXPLORING = "exploring"
    RETURNING_TO_NEST = "returning_to_nest"
    FOLLOWING_PHEROMONE = "following_pheromone"
    COLLECTING_FOOD = "collecting_food"

class PheromoneType(Enum):
    FOOD = "food"
    NEST = "nest"

@dataclass
class PheromoneLayer:
    width: int
    height: int
    evaporation_rate: float = 0.1
    diffusion_rate: float = 0.1

    def __post_init__(self):
        self.grid = np.zeros((self.height, self.width))

    def add_pheromone(self, x: int, y: int, amount: float):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] += amount

    def get_pheromone(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x]
        return 0.0

    def update(self):
        # Evaporation
        self.grid *= (1 - self.evaporation_rate)

        # Diffusion
        if self.diffusion_rate > 0:
            new_grid = self.grid.copy()
            for y in range(1, self.height-1):
                for x in range(1, self.width-1):
                    diffusion = self.grid[y-1:y+2, x-1:x+2].sum() * self.diffusion_rate / 8
                    new_grid[y, x] += diffusion
            self.grid = new_grid

@dataclass
class AntMemory:
    maze_width: int
    maze_height: int
    visited_positions: Set[Tuple[int, int]] = field(default_factory=set)
    pheromone_trail: List[Tuple[int, int]] = field(default_factory=list)
    nest_position: Tuple[int, int] = (2, 2)  # Fixed position
    exploration_bias: float = 0.7  # Bias towards unexplored areas

    def __post_init__(self):
        self.visited_positions = set()
        self.pheromone_trail = []

    def remember_position(self, x: int, y: int):
        """Remember a visited position"""
        if 0 <= x < self.maze_width and 0 <= y < self.maze_height:
            self.visited_positions.add((x, y))

    def is_position_visited(self, x: int, y: int) -> bool:
        """Check if a position has been visited"""
        return (x, y) in self.visited_positions

    def add_to_trail(self, x: int, y: int):
        """Add position to pheromone trail"""
        pos = (x, y)
        if pos not in self.pheromone_trail:
            self.pheromone_trail.append(pos)

class Agent:
    def __init__(self, x: int, y: int, maze_width: int, maze_height: int):
        self.x = x
        self.y = y
        self.state = AntState.EXPLORING
        self.memory = AntMemory(maze_width, maze_height)
        self.carrying_food = False
        self.food_amount = 0.0
        self.pheromone_sensitivity = 0.8  # Increased from original
        self.food_capacity = 5.0
        self.exploration_radius = 3  # How far the ant can "see"
        self.last_positions: List[Tuple[int, int]] = []  # Prevent immediate backtracking

    def decide_move(self, surroundings: Dict[Tuple[int, int], float],
                   pheromones: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        if self.state == AntState.EXPLORING:
            return self._explore(surroundings, pheromones)
        elif self.state == AntState.RETURNING_TO_NEST:
            return self._return_to_nest(surroundings)
        elif self.state == AntState.FOLLOWING_PHEROMONE:
            return self._follow_pheromone(surroundings, pheromones)
        return None

    def _get_possible_moves(self, surroundings: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        """Get all possible moves, excluding walls"""
        moves = []
        for (dx, dy), value in surroundings.items():
            if value != 1.0:  # Not a wall
                new_x, new_y = self.x + dx, self.y + dy
                if (new_x, new_y) not in self.last_positions[-3:]:  # Prevent recent backtracking
                    moves.append((new_x, new_y))
        return moves

    def _explore(self, surroundings: Dict[Tuple[int, int], float],
                pheromones: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """Improved exploration strategy"""
        possible_moves = self._get_possible_moves(surroundings)
        if not possible_moves:
            return None

        # Separate moves into unvisited and visited
        unvisited_moves = []
        pheromone_moves = []

        for new_pos in possible_moves:
            new_x, new_y = new_pos
            if not self.memory.is_position_visited(new_x, new_y):
                unvisited_moves.append(new_pos)

            # Check pheromone strength
            dx, dy = new_x - self.x, new_y - self.y
            pheromone_value = pheromones.get((dx, dy), 0.0)
            if pheromone_value > 0:
                pheromone_moves.append((new_pos, pheromone_value))

        # Probabilistic decision making
        if pheromone_moves and random.random() < self.pheromone_sensitivity:
            # Follow pheromones with probability based on strength
            total_pheromone = sum(p[1] for p in pheromone_moves)
            weights = [p[1]/total_pheromone for p in pheromone_moves]
            chosen_pos = random.choices([p[0] for p in pheromone_moves], weights=weights)[0]
            return chosen_pos

        # Prefer unvisited positions with high probability
        if unvisited_moves and random.random() < self.memory.exploration_bias:
            return random.choice(unvisited_moves)

        # Otherwise, choose any valid move
        return random.choice(possible_moves)

    def _follow_pheromone(self, surroundings: Dict[Tuple[int, int], float],
                         pheromones: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """Follow pheromone trails with probability based on pheromone strength"""
        # Get possible moves with their pheromone values
        pheromone_moves = []

        for (dx, dy), value in surroundings.items():
            if value != 1.0:  # Not a wall
                new_pos = (self.x + dx, self.y + dy)
                pheromone_value = pheromones.get((dx, dy), 0.0)

                # Only consider moves with pheromone
                if pheromone_value > 0:
                    pheromone_moves.append((new_pos, pheromone_value))

        # If we have pheromone trails to follow
        if pheromone_moves:
            # Calculate probability weights based on pheromone strength
            total_pheromone = sum(p[1] for p in pheromone_moves)
            weights = [p[1]/total_pheromone for p in pheromone_moves]

            # Choose move based on pheromone weights
            return random.choices([p[0] for p in pheromone_moves], weights=weights)[0]

        # If no pheromone trails found, switch back to exploring
        self.state = AntState.EXPLORING
        return self._explore(surroundings, pheromones)

    def _return_to_nest(self, surroundings: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        if self.memory.pheromone_trail:
            next_pos = self.memory.pheromone_trail[-1]
            # Verify move is possible
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            if surroundings.get((dx, dy), 1.0) != 1.0:  # Not a wall
                self.memory.pheromone_trail.pop()
                return next_pos
        return self._navigate_to(self.memory.nest_position, surroundings)

    def _navigate_to(self, target: Tuple[int, int],
                    surroundings: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        tx, ty = target
        dx = tx - self.x
        dy = ty - self.y

        # Try to move in the direction of the target
        possible_moves = []
        if abs(dx) > abs(dy):
            # Try horizontal first
            if dx != 0:
                move = (1 if dx > 0 else -1, 0)
                if surroundings.get(move, 1.0) != 1.0:
                    possible_moves.append((self.x + move[0], self.y + move[1]))
            # Then vertical
            if dy != 0:
                move = (0, 1 if dy > 0 else -1)
                if surroundings.get(move, 1.0) != 1.0:
                    possible_moves.append((self.x + move[0], self.y + move[1]))
        else:
            # Try vertical first
            if dy != 0:
                move = (0, 1 if dy > 0 else -1)
                if surroundings.get(move, 1.0) != 1.0:
                    possible_moves.append((self.x + move[0], self.y + move[1]))
            # Then horizontal
            if dx != 0:
                move = (1 if dx > 0 else -1, 0)
                if surroundings.get(move, 1.0) != 1.0:
                    possible_moves.append((self.x + move[0], self.y + move[1]))

        if possible_moves:
            return random.choice(possible_moves)
        return None

    def update(self, maze_info: Dict):
        """Update agent state and position"""
        self.memory.remember_position(self.x, self.y)

        # Handle food collection
        if maze_info.get('food_here', False) and not self.carrying_food:
            food_amount = maze_info.get('food_amount', 0.0)
            if food_amount > 0:
                self.collect_food(min(self.food_capacity, food_amount))
                self.state = AntState.RETURNING_TO_NEST

        # Handle nest return
        if self.carrying_food and (self.x, self.y) == self.memory.nest_position:
            self.deposit_food()
            self.state = AntState.EXPLORING
            self.last_positions.clear()  # Reset position history after reaching nest

        # Update position
        next_pos = self.decide_move(
            maze_info['surroundings'],
            maze_info.get('pheromones', {})
        )

        if next_pos:
            self.x, self.y = next_pos
            self.last_positions.append((self.x, self.y))
            if len(self.last_positions) > 10:  # Keep last 10 positions
                self.last_positions.pop(0)

            if self.carrying_food:
                self.memory.add_to_trail(self.x, self.y)

    def collect_food(self, amount: float):
        """Collect food from source"""
        self.carrying_food = True
        self.food_amount = amount
        self.state = AntState.RETURNING_TO_NEST

    def deposit_food(self) -> float:
        """Deposit food at nest"""
        amount = self.food_amount
        self.carrying_food = False
        self.food_amount = 0
        self.memory.pheromone_trail.clear()
        return amount
