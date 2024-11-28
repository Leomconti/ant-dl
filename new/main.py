import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import pygame

CELL_SIZE = 10  # px
ANT_SIZE = CELL_SIZE
VISION_RANGE = 2
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
        self.x = int(x)
        self.y = int(y)
        self.state = AntState.EXPLORING
        self.carrying_food = False
        self.target_food: Optional[Food] = None
        self.env = environment
        self.memory: List[Tuple[int, int]] = []
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.known_walls: Set[Tuple[int, int]] = set()

    def get_vision_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x1 = max(0, int(self.x - VISION_RANGE))
        y1 = max(0, int(self.y - VISION_RANGE))
        x2 = min(self.env.width - 1, int(self.x + VISION_RANGE + 1))
        y2 = min(self.env.height - 1, int(self.y + VISION_RANGE + 1))
        return ((x1, y1), (x2, y2))

    def find_valid_direction(self) -> Tuple[int, int]:
        # Get current vision bounds
        (x1, y1), (x2, y2) = self.get_vision_bounds()
        current_cell = (int(self.x), int(self.y))

        # List all possible moves (adjacent cells)
        possible_moves = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_x = int(self.x) + dx
            new_y = int(self.y) + dy
            new_cell = (new_x, new_y)

            # Check if move is valid
            if (
                x1 <= new_x <= x2
                and y1 <= new_y <= y2
                and new_cell not in self.env.walls
                and new_cell not in self.known_walls
                and new_cell not in self.memory[-3:]
                and new_cell != current_cell
            ):
                possible_moves.append((dx, dy))

        if not possible_moves:
            return (random.randint(-1, 1), random.randint(-1, 1))

        # If returning to nest or going to food, choose move closest to target
        if self.state == AntState.RETURNING:
            target = self.env.nest_location
        elif self.target_food:
            target = (self.target_food.x, self.target_food.y)
        else:
            # Choose random valid move, preferring unvisited cells
            move = random.choice(possible_moves)
            return move

        # Find move that gets us closest to target
        best_move = min(
            possible_moves,
            key=lambda m: ((int(self.x) + m[0] - target[0]) ** 2 + (int(self.y) + m[1] - target[1]) ** 2),
        )
        return best_move

    def move(self, next_x: int, next_y: int):
        if (next_x, next_y) not in self.env.walls:
            if 0 <= next_x < self.env.width and 0 <= next_y < self.env.height:
                self.x = next_x
                self.y = next_y
                current_pos = (self.x, self.y)
                self.visited_cells.add(current_pos)
                if current_pos not in self.memory:
                    self.memory.append(current_pos)
                    if len(self.memory) > 50:
                        self.memory.pop(0)

    def explore(self):
        # Check for nearby food
        for food in self.env.food:
            if food.value > 0 and abs(food.x - self.x) < VISION_RANGE and abs(food.y - self.y) < VISION_RANGE:
                self.target_food = food
                break

        # Get next position
        dx, dy = self.find_valid_direction()
        next_x = self.x + dx
        next_y = self.y + dy

        # Move one cell
        self.move(next_x, next_y)

        # Check if reached food
        for food in self.env.food:
            if food.value > 0 and self.x == food.x and self.y == food.y:
                self.carrying_food = True
                self.state = AntState.RETURNING
                self.target_food = food
                food.value -= 1
                self.memory = []
                break

    def return_to_nest(self):
        dx, dy = self.find_valid_direction()
        next_x = self.x + dx
        next_y = self.y + dy

        # Update pheromone at current position
        pos = (self.x, self.y)
        current_strength = self.env.pheromone_layer.get(pos, 0)
        self.env.pheromone_layer[pos] = min(MAX_PHEROMONE, current_strength + PHEROMONE_DEPOSIT)

        self.move(next_x, next_y)

        # Check if reached nest
        if self.x == self.env.nest_location[0] and self.y == self.env.nest_location[1]:
            self.carrying_food = False
            self.env.nest_food += 1
            if self.target_food and self.target_food.value > 0:
                self.state = AntState.EXPLORING
                self.memory = []
            else:
                self.state = AntState.EXPLORING
                self.target_food = None
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
        self.nest_location = (2, 2)
        self.nest_food = 0

        # Initialize Pygame surfaces
        self.screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
        self.wall_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.pheromone_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.vision_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)
        self.entity_surface = pygame.Surface((width * CELL_SIZE, height * CELL_SIZE), pygame.SRCALPHA)

        # Initialize font for food values
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 10)

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
        adjacent_walls = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if (x + dx, y + dy) in self.walls)
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
                self.wall_surface, (100, 100, 100), (wall_x * CELL_SIZE, wall_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

        # Draw pheromone trails
        for (x, y), strength in self.pheromone_layer.items():
            color_value = min(255, int(strength))
            pygame.draw.rect(
                self.pheromone_surface,
                (0, color_value, 0, color_value),  # Green with alpha
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw ant vision (optional)
        for ant in self.ants:
            (x1, y1), (x2, y2) = ant.get_vision_bounds()
            pygame.draw.rect(
                self.vision_surface,
                (200, 200, 0, 50),
                (x1 * CELL_SIZE, y1 * CELL_SIZE, (x2 - x1) * CELL_SIZE, (y2 - y1) * CELL_SIZE),
            )

        # Draw nest with food value
        pygame.draw.rect(
            self.entity_surface,
            (255, 192, 203),  # Pink
            (self.nest_location[0] * CELL_SIZE, self.nest_location[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )
        # Add these lines to display nest food value
        nest_text = self.font.render(str(self.nest_food), True, (0, 0, 0))
        text_rect = nest_text.get_rect(
            center=(
                self.nest_location[0] * CELL_SIZE + CELL_SIZE // 2,
                self.nest_location[1] * CELL_SIZE + CELL_SIZE // 2,
            )
        )
        self.entity_surface.blit(nest_text, text_rect)

        # Draw food with values - for how much food there is in the cell
        for food in self.food:
            if food.value > 0:
                # Draw food
                pygame.draw.rect(
                    self.entity_surface, (255, 0, 0), (food.x * CELL_SIZE, food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
                # Draw food value
                value_text = self.font.render(str(food.value), True, (255, 255, 255))
                text_rect = value_text.get_rect(
                    center=(food.x * CELL_SIZE + CELL_SIZE // 2, food.y * CELL_SIZE + CELL_SIZE // 2)
                )
                self.entity_surface.blit(value_text, text_rect)

        # Draw ants
        for ant in self.ants:
            # Blue for exploring, orange for returning
            color = (0, 0, 255) if ant.state == AntState.EXPLORING else (255, 165, 0)
            pygame.draw.rect(
                self.entity_surface,
                color,
                (ant.x * CELL_SIZE - ANT_SIZE / 2, ant.y * CELL_SIZE - ANT_SIZE / 2, ANT_SIZE, ANT_SIZE),
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
    env = Environment(40, 40)
    env.create_maze()

    # Add some food sources
    food_positions = [(10, 10), (20, 20), (30, 30)]
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
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
