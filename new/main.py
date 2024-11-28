import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Tuple

import pygame

SURROUNDINGS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

CELL_SIZE = 10  # px
ANT_SIZE = CELL_SIZE
VISION_RANGE = 2
MAX_PHEROMONE = 255
PHEROMONE_DEPOSIT = 20
PHEROMONE_EVAPORATION = 0.998
MOVE_SPEED = 0.5


class AntState(Enum):
    EXPLORING = 1
    RETURNING = 2
    BACK_TO_FOOD = 3


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
        self.path_to_food: List[Tuple[int, int]] = []

    def get_vision_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x1 = max(0, int(self.x - VISION_RANGE))
        y1 = max(0, int(self.y - VISION_RANGE))
        x2 = min(self.env.width - 1, int(self.x + VISION_RANGE + 1))
        y2 = min(self.env.height - 1, int(self.y + VISION_RANGE + 1))
        return ((x1, y1), (x2, y2))

    def _check_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.env.width and 0 <= y < self.env.height and (x, y) not in self.env.walls

    def move(self, next_x: int, next_y: int):
        if next_x == self.x and next_y == self.y:
            return  # Do not move if the next position is the same
        if self._check_bounds(next_x, next_y):
            self.x = next_x
            self.y = next_y
            current_pos = (self.x, self.y)
            self.visited_cells.add(current_pos)
            self.memory.append(current_pos)
        else:
            print(f"Out of bounds: {next_x}, {next_y}")

    def _food_in_range(self) -> Literal["HERE", "NEARBY", None]:
        # First check if we're directly on top of food
        for food in self.env.food:
            if food.value > 0 and self.x == food.x and self.y == food.y:
                self.target_food = food  # Store the found food
                return "HERE"

        # Then check vision range if we're not on top of any food
        closest_food = None
        closest_dist = float("inf")

        for food in self.env.food:
            if food.value > 0:
                dist = abs(food.x - self.x) + abs(food.y - self.y)
                if dist <= VISION_RANGE and dist < closest_dist:
                    closest_food = food
                    closest_dist = dist

        if closest_food:
            self.target_food = closest_food  # Store the found food
            return "NEARBY"

        return None

    def find_valid_direction(self) -> Tuple[int, int]:
        valid_directions = []
        for dx, dy in SURROUNDINGS:
            if self._check_bounds(self.x + dx, self.y + dy):
                valid_directions.append((dx, dy))
        if valid_directions:
            return random.choice(valid_directions)
        return 0, 0

    def collect_food(self):
        for food in self.env.food:
            if food.value > 0 and self.x == food.x and self.y == food.y:
                self.carrying_food = True
                food.value -= 1
                self.state = AntState.RETURNING
                return
        raise ValueError("No food found, why did you collect???")

    def _get_closest_food(self) -> Optional[Food]:
        return self.target_food  # Simply return the stored target food

    def _find_best_direction_to_explore(self) -> Tuple[int, int]:
        best_direction = None
        highest_pheromone = -1
        for dx, dy in SURROUNDINGS:
            next_x = self.x + dx
            next_y = self.y + dy
            if not self._check_bounds(next_x, next_y):
                continue
            pheromone = self.env.pheromone_layer.get((next_x, next_y), 0)
            if pheromone > highest_pheromone:
                highest_pheromone = pheromone
                best_direction = (dx, dy)
        if best_direction is not None and highest_pheromone > 0:
            return best_direction
        # If no pheromone nearby, move randomly
        return self.find_valid_direction()

    def explore(self):
        food_range = self._food_in_range()
        if food_range is not None:
            if food_range == "HERE":
                self.collect_food()
                return
            elif food_range == "NEARBY" and self.target_food:
                dx = max(-1, min(1, self.target_food.x - self.x))
                dy = max(-1, min(1, self.target_food.y - self.y))
                next_x = self.x + dx
                next_y = self.y + dy
                self.move(next_x, next_y)
                return
        # No food in range - Follow pheromone trails
        dx, dy = self._find_best_direction_to_explore()
        next_x = self.x + dx
        next_y = self.y + dy
        self.move(next_x, next_y)

    def _reached_nest(self) -> bool:
        return self.x == self.env.nest_location[0] and self.y == self.env.nest_location[1]

    def _find_best_direction_to_nest(self) -> Tuple[int, int]:
        # First try to use memory to backtrack
        if self.memory:
            # Get the last few positions and find one that's closer to the nest
            for prev_pos in reversed(self.memory[-10:]):  # Look at last 10 positions
                dx = prev_pos[0] - self.x
                dy = prev_pos[1] - self.y
                if dx == 0 and dy == 0:
                    continue  # Skip if it doesn't result in movement
                if abs(dx) <= 1 and abs(dy) <= 1:  # If it's an adjacent cell
                    if self._check_bounds(self.x + dx, self.y + dy):
                        return dx, dy

        # If memory doesn't help, use pheromones
        best_direction = (0, 0)
        highest_pheromone = -1
        current_dist_to_nest = abs(self.x - self.env.nest_location[0]) + abs(self.y - self.env.nest_location[1])

        for dx, dy in SURROUNDINGS:
            next_x = self.x + dx
            next_y = self.y + dy

            if not self._check_bounds(next_x, next_y):
                continue

            # Calculate if this move brings us closer to the nest
            new_dist_to_nest = abs(next_x - self.env.nest_location[0]) + abs(next_y - self.env.nest_location[1])
            closer_to_nest = new_dist_to_nest < current_dist_to_nest

            # Check pheromone level at this position
            pheromone = self.env.pheromone_layer.get((next_x, next_y), 0)

            # Prefer positions that are closer to nest and have higher pheromone
            if closer_to_nest and pheromone > highest_pheromone:
                highest_pheromone = pheromone
                best_direction = (dx, dy)

        # If no good direction found, move directly towards the nest
        if best_direction == (0, 0):
            dx = max(-1, min(1, self.env.nest_location[0] - self.x))
            dy = max(-1, min(1, self.env.nest_location[1] - self.y))
            if self._check_bounds(self.x + dx, self.y + dy):
                return dx, dy
        # If direct path is blocked, move randomly
        return self.find_valid_direction()

    def update_pheromones(self):
        pos = (self.x, self.y)
        current_strength = self.env.pheromone_layer.get(pos, 0)
        deposit_amount = PHEROMONE_DEPOSIT if self.carrying_food else PHEROMONE_DEPOSIT / 2
        self.env.pheromone_layer[pos] = min(MAX_PHEROMONE, current_strength + deposit_amount)

    def return_to_nest(self):
        # Update pheromone at current position
        self.update_pheromones()
        # Find best direction to nest using pheromones and memory
        dx, dy = self._find_best_direction_to_nest()
        next_x = self.x + dx
        next_y = self.y + dy

        # Move one cell closer to nest
        self.move(next_x, next_y)

        # Check if reached nest
        if self._reached_nest():
            self.carrying_food = False
            self.env.nest_food += 1
            # If there's still food, go back to it using stored path
            if self.target_food and self.target_food.value > 0:
                self.state = AntState.BACK_TO_FOOD
                self.path_to_food.reverse()  # Reverse path to follow it back
            else:
                self.state = AntState.EXPLORING
                self.target_food = None
                self.path_to_food.clear()

    def back_to_food(self):
        if not self.path_to_food:
            # If we've lost the path, go back to exploring
            self.state = AntState.EXPLORING
            return

        next_pos = self.path_to_food.pop()
        self.move(next_pos[0], next_pos[1])

        # Check if we've reached the food
        if self.target_food and self.x == self.target_food.x and self.y == self.target_food.y:
            if self.target_food.value > 0:
                self.carrying_food = True
                self.target_food.value -= 1
                self.state = AntState.RETURNING
                self.path_to_food.clear()  # Clear the path as we'll make a new one
            else:
                self.state = AntState.EXPLORING
                self.target_food = None
                self.path_to_food.clear()

    def update(self):
        if self.state == AntState.EXPLORING:
            self.explore()
        elif self.state == AntState.RETURNING:
            self.return_to_nest()
        elif self.state == AntState.BACK_TO_FOOD:
            self.back_to_food()


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

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 10)

    def create_maze(self):
        # Create outer walls only
        for x in range(self.width):
            self.walls.add((x, 0))
            self.walls.add((x, self.height - 1))
        for y in range(self.height):
            self.walls.add((0, y))
            self.walls.add((self.width - 1, y))

        # Ensure nest location is clear (although it shouldn't be a wall anyway)
        self.walls.discard(self.nest_location)

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
        """
        Add ant to nest + env
        """
        ant = Ant(self.nest_location[0], self.nest_location[1], self)
        self.ants.append(ant)

    def update(self):
        """
        Update environment things.
        - Pheromone evaporation
        - And update
        """
        for pos in list(self.pheromone_layer.keys()):
            self.pheromone_layer[pos] *= PHEROMONE_EVAPORATION
            if self.pheromone_layer[pos] < 1:
                del self.pheromone_layer[pos]

        for ant in self.ants:
            ant.update()

    def render(self):
        self.screen.fill((255, 255, 255))
        # CLEAR
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
        self.screen.blit(self.pheromone_surface, (0, 0))  # pheromone layer - render pheromone trails
        self.screen.blit(self.wall_surface, (0, 0))  # wall layer - show walls and floor
        self.screen.blit(self.vision_surface, (0, 0))  # vision layer - render ant vision
        self.screen.blit(self.entity_surface, (0, 0))  # entity layer - render nest, food and ants

        pygame.display.flip()


def main():
    pygame.init()

    # Create environment
    env = Environment(50, 50)
    env.create_maze()

    # Add some food sources
    food_positions = [(20, 20)]
    for pos in food_positions:
        env.add_food(*pos)

    # Add some ants
    for _ in range(40):
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
