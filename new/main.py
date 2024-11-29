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


@dataclass
class Pheromone:
    x: int
    y: int
    strength: float
    food_direction: Tuple[int, int]

    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def evaporate(self) -> None:
        self.strength *= PHEROMONE_EVAPORATION


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
        self.memory = []
        self.path_home = []
        self.last_food_location = None
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
        # Random exploration chance to prevent getting stuck
        if random.random() < 0.1:
            return self.find_valid_direction()

        best_direction = None
        highest_pheromone = -1
        valid_moves = []

        for dx, dy in SURROUNDINGS:
            next_x = self.x + dx
            next_y = self.y + dy
            if not self._check_bounds(next_x, next_y):
                continue

            next_pos = (next_x, next_y)
            # Add weight to unvisited cells
            if next_pos not in self.visited_cells:
                valid_moves.append((dx, dy))

            if next_pos in self.env.pheromone_layer:
                pheromone = self.env.pheromone_layer[next_pos]
                # Add some randomness to pheromone following
                random_factor = random.uniform(0.8, 1.2)
                effective_strength = pheromone.strength * random_factor

                if effective_strength > highest_pheromone:
                    highest_pheromone = effective_strength
                    best_direction = pheromone.food_direction

        # If we found a good pheromone trail, follow it
        if best_direction is not None and highest_pheromone > 20:
            return best_direction

        # If we have valid unvisited moves, prefer those
        if valid_moves:
            return random.choice(valid_moves)

        # Otherwise, move randomly
        return self.find_valid_direction()

    def _find_path_to_target(self, target_x: int, target_y: int) -> List[Tuple[int, int]]:
        """New: Simple A* pathfinding to target"""
        from queue import PriorityQueue

        def heuristic(x1, y1):
            return abs(x1 - target_x) + abs(y1 - target_y)

        frontier = PriorityQueue()
        frontier.put((0, (self.x, self.y)))
        came_from = {(self.x, self.y): None}
        cost_so_far = {(self.x, self.y): 0}

        while not frontier.empty():
            current = frontier.get()[1]

            if current == (target_x, target_y):
                break

            for dx, dy in SURROUNDINGS:
                next_pos = (current[0] + dx, current[1] + dy)
                if not self._check_bounds(next_pos[0], next_pos[1]):
                    continue

                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos[0], next_pos[1])
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        path = []
        current = (target_x, target_y)
        while current in came_from and came_from[current] is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    def explore(self):
        """Improved exploring behavior"""
        food_status = self._food_in_range()

        if food_status == "HERE":
            self.last_food_location = (self.x, self.y)
            self.path_home = self._find_path_to_target(self.env.nest_location[0], self.env.nest_location[1])
            self.collect_food()
            return

        if food_status == "NEARBY" and self.target_food:
            # Path to nearby food
            path = self._find_path_to_target(self.target_food.x, self.target_food.y)
            if path:
                next_pos = path[0]
                self.move(next_pos[0], next_pos[1])
                return

        # Follow pheromones or explore randomly
        best_direction = self._find_best_direction_to_explore()
        next_x = self.x + best_direction[0]
        next_y = self.y + best_direction[1]
        self.move(next_x, next_y)

    def _reached_nest(self) -> bool:
        return self.x == self.env.nest_location[0] and self.y == self.env.nest_location[1]

    def _find_best_direction_to_nest(self) -> Tuple[int, int]:
        # Move directly towards the nest
        dx = max(-1, min(1, self.env.nest_location[0] - self.x))
        dy = max(-1, min(1, self.env.nest_location[1] - self.y))

        # If direct path is available, use it
        if self._check_bounds(self.x + dx, self.y + dy):
            return dx, dy

        # If direct path is blocked, try horizontal or vertical movement
        if dx != 0 and self._check_bounds(self.x + dx, self.y):
            return dx, 0
        if dy != 0 and self._check_bounds(self.x, self.y + dy):
            return 0, dy

        # If all direct paths are blocked, move randomly
        return self.find_valid_direction()

    def update_pheromones(self):
        pos = (self.x, self.y)
        if not self.memory:
            return

        # Get direction from previous position
        prev_pos = self.memory[-1] if len(self.memory) > 1 else pos
        direction = (max(min(self.x - prev_pos[0], 1), -1), max(min(self.y - prev_pos[1], 1), -1))

        # Ensure direction is one of the valid surrounding positions
        if direction not in SURROUNDINGS:
            direction = (0, 0)  # Default to no direction if invalid

        deposit_amount = PHEROMONE_DEPOSIT if self.carrying_food else PHEROMONE_DEPOSIT / 2

        # Get existing pheromone or create new one
        if pos in self.env.pheromone_layer:
            current_strength = self.env.pheromone_layer[pos].strength
        else:
            current_strength = 0

        new_strength = min(MAX_PHEROMONE, current_strength + deposit_amount)

        # Create new pheromone
        self.env.pheromone_layer[pos] = Pheromone(x=self.x, y=self.y, strength=new_strength, food_direction=direction)

    def return_to_nest(self):
        """Improved return behavior"""
        self.update_pheromones()
        if not self.path_home:
            self.path_home = self._find_path_to_target(self.env.nest_location[0], self.env.nest_location[1])

        if self.path_home:
            next_pos = self.path_home[0]
            self.path_home = self.path_home[1:]
            self.move(next_pos[0], next_pos[1])

        if self._reached_nest():
            self.carrying_food = False
            self.env.nest_food += 1

            # If we remember where we found food and it still has value, go back
            if self.last_food_location and any(
                f.value > 0 and f.x == self.last_food_location[0] and f.y == self.last_food_location[1]
                for f in self.env.food
            ):
                self.state = AntState.BACK_TO_FOOD
                self.path_to_food = self._find_path_to_target(self.last_food_location[0], self.last_food_location[1])
            else:
                self.state = AntState.EXPLORING
                self.target_food = None
                self.last_food_location = None

            self.path_home = []

    def back_to_food(self):
        """Improved return to food behavior"""
        if not self.path_to_food:
            self.state = AntState.EXPLORING
            return

        next_pos = self.path_to_food[0]
        self.path_to_food = self.path_to_food[1:]
        self.move(next_pos[0], next_pos[1])

        # Check if we reached the food location
        if self.last_food_location and (self.x, self.y) == self.last_food_location:
            food = next((f for f in self.env.food if f.x == self.x and f.y == self.y and f.value > 0), None)
            if food:
                self.carrying_food = True
                food.value -= 1
                self.state = AntState.RETURNING
                self.path_home = self._find_path_to_target(self.env.nest_location[0], self.env.nest_location[1])
            else:
                self.state = AntState.EXPLORING
                self.last_food_location = None

    def update(self):
        # Add pheromone update at the start of every update
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
        self.pheromone_layer: Dict[Tuple[int, int], Pheromone] = {}
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
        """Update environment things."""
        # Update pheromone evaporation
        for pos in list(self.pheromone_layer.keys()):
            pheromone = self.pheromone_layer[pos]
            pheromone.evaporate()
            if pheromone.strength < 1:
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
        for pheromone in self.pheromone_layer.values():
            color_value = min(255, int(pheromone.strength))
            cell_rect = pygame.Rect(pheromone.x * CELL_SIZE, pheromone.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            pygame.draw.rect(
                self.pheromone_surface,
                (0, color_value, 0, color_value),
                cell_rect,
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
