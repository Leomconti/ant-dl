import pygame
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from enum import Enum
import numpy as np

# Constants
CELL_SIZE = 10
ANT_SIZE = CELL_SIZE
VISION_RANGE = 2  # This creates a 5x5 vision field
MAX_PHEROMONE = 255
PHEROMONE_DEPOSIT = 10
PHEROMONE_EVAPORATION = 0.995
MOVE_SPEED = 0.5

class AntState(Enum):
    EXPLORING = 1
    RETURNING = 2

@dataclass
class Food:
    x: int
    y: int
    value: int

class Ant:
    def __init__(self, x: float, y: float, environment):
        self.x = float(x)
        self.y = float(y)
        self.direction = random.uniform(0, 2 * np.pi)
        self.state = AntState.EXPLORING
        self.carrying_food = False
        self.target_food: Optional[Food] = None
        self.env = environment
        self.memory: List[Tuple[int, int]] = []  # Short-term memory to avoid getting stuck
        self.visited_cells: Set[Tuple[int, int]] = set()  # Long-term memory of visited cells
        self.known_walls: Set[Tuple[int, int]] = set()    # Memory of known walls

    def get_vision_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x1 = max(0, int(self.x - VISION_RANGE))
        y1 = max(0, int(self.y - VISION_RANGE))
        x2 = min(self.env.width - 1, int(self.x + VISION_RANGE + 1))
        y2 = min(self.env.height - 1, int(self.y + VISION_RANGE + 1))
        return ((x1, y1), (x2, y2))

    def get_next_position(self, direction: float) -> Tuple[float, float]:
        next_x = self.x + np.cos(direction) * MOVE_SPEED
        next_y = self.y + np.sin(direction) * MOVE_SPEED
        return next_x, next_y  # Avoid rounding

    def wall_ahead(self, next_x: float, next_y: float) -> bool:
        # Improved collision detection along the path
        x0, y0 = self.x, self.y
        x1, y1 = next_x, next_y
        distance = np.hypot(x1 - x0, y1 - y0)
        steps = int(distance / 0.1) + 1  # Number of steps

        for i in range(1, steps + 1):
            t = i / steps
            xi = x0 + (x1 - x0) * t
            yi = y0 + (y1 - y0) * t
            cell = (int(xi), int(yi))
            if cell in self.env.walls:
                self.known_walls.add(cell)  # Remember this wall
                return True
        return False

    def find_valid_direction(self) -> float:
        # Try 16 different directions
        possible_directions = []
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            test_direction = angle
            next_x, next_y = self.get_next_position(test_direction)
            cell = (int(next_x), int(next_y))
            if cell in self.known_walls:
                continue  # Skip known walls
            if not self.wall_ahead(next_x, next_y):
                possible_directions.append((test_direction, cell))

        if not possible_directions:
            # If no valid directions, try random directions
            while True:
                random_direction = random.uniform(0, 2 * np.pi)
                next_x, next_y = self.get_next_position(random_direction)
                cell = (int(next_x), int(next_y))
                if cell not in self.known_walls and not self.wall_ahead(next_x, next_y):
                    return random_direction

        # Prioritize unvisited cells
        unvisited_directions = [ (d, c) for d, c in possible_directions if c not in self.visited_cells ]
        if unvisited_directions:
            possible_directions = unvisited_directions

        # If we have a specific target (food or nest)
        if self.state == AntState.RETURNING:
            target_x, target_y = self.env.nest_location
        elif self.target_food:
            target_x, target_y = self.target_food.x, self.target_food.y
        else:
            # If no target, choose a random direction among unvisited cells
            return random.choice([d for d, c in possible_directions])

        # Find direction closest to target while avoiding known walls and visited cells
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = np.arctan2(dy, dx)

        # Sort directions by closeness to target angle
        possible_directions.sort(key=lambda dc: abs(np.arctan2(np.sin(dc[0] - target_angle), np.cos(dc[0] - target_angle))))

        # Try directions in order of preference
        for direction, cell in possible_directions:
            if cell not in self.visited_cells:
                return direction

        # If all directions are visited, choose the best one
        return possible_directions[0][0]

    def move(self, next_x: float, next_y: float):
        if not self.wall_ahead(next_x, next_y):
            if 0 <= next_x < self.env.width and 0 <= next_y < self.env.height:
                self.x = next_x
                self.y = next_y
                current_pos = (int(self.x), int(self.y))
                self.visited_cells.add(current_pos)  # Record visited cell
                if current_pos not in self.memory:
                    self.memory.append(current_pos)
                    if len(self.memory) > 50:
                        self.memory.pop(0)
            else:
                # If next position is outside the environment, adjust direction
                self.direction = self.find_valid_direction()
        else:
            # If wall ahead, adjust direction
            self.direction = self.find_valid_direction()

    def explore(self):
        # Check for nearby food
        for food in self.env.food:
            if (food.value > 0 and
                abs(food.x - self.x) < VISION_RANGE and
                abs(food.y - self.y) < VISION_RANGE):
                self.target_food = food
                dx = food.x - self.x
                dy = food.y - self.y
                self.direction = np.arctan2(dy, dx)
                break

        next_x, next_y = self.get_next_position(self.direction)

        if self.wall_ahead(next_x, next_y) or (int(next_x), int(next_y)) in self.visited_cells:
            self.direction = self.find_valid_direction()
            next_x, next_y = self.get_next_position(self.direction)

        self.move(next_x, next_y)

        # Check if reached food
        for food in self.env.food:
            if (food.value > 0 and
                int(self.x) == food.x and
                int(self.y) == food.y):
                self.carrying_food = True
                self.state = AntState.RETURNING
                self.target_food = food
                food.value -= 1
                self.memory = []  # Clear memory for return journey
                break

    def return_to_nest(self):
        dx = self.env.nest_location[0] - self.x
        dy = self.env.nest_location[1] - self.y
        distance_to_nest = np.sqrt(dx * dx + dy * dy)

        self.direction = np.arctan2(dy, dx)
        next_x, next_y = self.get_next_position(self.direction)

        if self.wall_ahead(next_x, next_y) or (int(next_x), int(next_y)) in self.known_walls:
            # Adjust direction to avoid walls
            self.direction = self.find_valid_direction()
            next_x, next_y = self.get_next_position(self.direction)

        # Update pheromone
        pos = (int(self.x), int(self.y))
        current_strength = self.env.pheromone_layer.get(pos, 0)
        self.env.pheromone_layer[pos] = min(
            MAX_PHEROMONE,
            current_strength + PHEROMONE_DEPOSIT
        )

        self.move(next_x, next_y)

        # Check if reached nest
        if (int(self.x) == self.env.nest_location[0] and
            int(self.y) == self.env.nest_location[1]):
            self.carrying_food = False
            if self.target_food and self.target_food.value > 0:
                self.state = AntState.EXPLORING
                self.memory = []  # Clear memory for new journey
            else:
                self.state = AntState.EXPLORING
                self.target_food = None
                self.direction = random.uniform(0, 2 * np.pi)
                self.memory = []

    def update(self):
        if self.state == AntState.EXPLORING:
            self.explore()
        elif self.state == AntState.RETURNING:
            self.return_to_nest()

class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: Set[Tuple[int, int]] = set()
        self.food: List[Food] = []
        self.ants: List[Ant] = []
        self.pheromone_layer: Dict[Tuple[int, int], float] = {}
        self.nest_location = (1, 1)

        # Initialize Pygame surfaces
        self.screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
        self.wall_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.pheromone_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.vision_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.entity_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)

        # Initialize font for food values
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 10)

    def create_maze(self):
        # Create outer walls
        for x in range(self.width):
            self.walls.add((x, 0))
            self.walls.add((x, self.height - 1))
        for y in range(self.height):
            self.walls.add((0, y))
            self.walls.add((self.width - 1, y))

        # Add some random walls, ensuring paths exist
        for _ in range(self.width * self.height // 10):
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)
            # Don't create walls that might trap ants
            if not self.would_block_path((x, y)):
                self.walls.add((x, y))

    def would_block_path(self, pos: Tuple[int, int]) -> bool:
        # Simple check to prevent creating walls that might trap ants
        x, y = pos
        adjacent_walls = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if (x + dx, y + dy) in self.walls)
        return adjacent_walls >= 2

    def add_food(self, x: int, y: int):
        if (x, y) not in self.walls and (x, y) != self.nest_location:
            food = Food(x, y, random.randint(50, 200))
            self.food.append(food)

    def add_ant(self):
        ant = Ant(self.nest_location[0], self.nest_location[1], self)
        self.ants.append(ant)

    def update(self):
        # Update pheromones with evaporation
        for pos in list(self.pheromone_layer.keys()):
            self.pheromone_layer[pos] *= PHEROMONE_EVAPORATION
            if self.pheromone_layer[pos] < 1:
                del self.pheromone_layer[pos]

        for ant in self.ants:
            ant.update()

    def render(self):
        self.screen.fill((255, 255, 255))

        # Clear surfaces
        self.wall_surface.fill((0, 0, 0, 0))
        self.pheromone_surface.fill((0, 0, 0, 0))
        self.vision_surface.fill((0, 0, 0, 0))
        self.entity_surface.fill((0, 0, 0, 0))

        # Draw walls
        for wall_x, wall_y in self.walls:
            pygame.draw.rect(
                self.wall_surface,
                (100, 100, 100),
                (wall_x * CELL_SIZE, wall_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

        # Draw pheromone trails
        for (x, y), strength in self.pheromone_layer.items():
            color_value = min(255, int(strength))
            pygame.draw.rect(
                self.pheromone_surface,
                (0, color_value, 0, color_value),  # Green with alpha
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

        # Draw ant vision (optional)
        for ant in self.ants:
            (x1, y1), (x2, y2) = ant.get_vision_bounds()
            pygame.draw.rect(
                self.vision_surface,
                (200, 200, 0, 50),
                (x1 * CELL_SIZE, y1 * CELL_SIZE,
                 (x2 - x1) * CELL_SIZE, (y2 - y1) * CELL_SIZE)
            )

        # Draw food and nest
        pygame.draw.rect(
            self.entity_surface,
            (0, 255, 0),
            (self.nest_location[0] * CELL_SIZE, self.nest_location[1] * CELL_SIZE,
             CELL_SIZE, CELL_SIZE)
        )

        # Draw food with values
        for food in self.food:
            if food.value > 0:
                pygame.draw.rect(
                    self.entity_surface,
                    (255, 0, 0),
                    (food.x * CELL_SIZE, food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
                # Draw food value
                value_text = self.font.render(str(food.value), True, (255, 255, 255))
                text_rect = value_text.get_rect(center=(
                    food.x * CELL_SIZE + CELL_SIZE // 2,
                    food.y * CELL_SIZE + CELL_SIZE // 2
                ))
                self.entity_surface.blit(value_text, text_rect)

        # Draw ants
        for ant in self.ants:
            color = (0, 0, 255) if ant.state == AntState.EXPLORING else (255, 165, 0)
            pygame.draw.rect(
                self.entity_surface,
                color,
                (ant.x * CELL_SIZE - ANT_SIZE / 2,
                 ant.y * CELL_SIZE - ANT_SIZE / 2,
                 ANT_SIZE, ANT_SIZE)
            )

        # Combine all layers
        self.screen.blit(self.pheromone_surface, (0, 0))
        self.screen.blit(self.wall_surface, (0, 0))
        self.screen.blit(self.vision_surface, (0, 0))
        self.screen.blit(self.entity_surface, (0, 0))

        pygame.display.flip()

def main():
    pygame.init()

    # Create environment
    env = Environment(80, 60)
    env.create_maze()

    # Add some food sources
    food_positions = [(60, 40), (20, 50), (70, 10)]
    for pos in food_positions:
        env.add_food(*pos)

    # Add some ants
    for _ in range(20):
        env.add_ant()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.update()
        env.render()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
