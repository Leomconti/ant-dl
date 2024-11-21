import pygame
import numpy as np
from typing import List, Tuple, Dict
import random
from agent import Agent
from consts import WALL_VALUE, EMPTY_BASE_VALUE, EMPTY_VARIATION, FOOD_INFLUENCE
from colors import Colors

class MazeEnvironment:
    def __init__(self, width: int, height: int, cell_size: int = 8):
        pygame.init()

        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Trabalho IA - Colonia de Formigas")

        # Initalize maze
        self.maze = np.zeros((height, width), dtype=float)
        self.initialize_maze_values()

        # Set the objects and component sof the maze
        self.agents: List[Agent] = []
        self.food_positions: List[Tuple[int, int]] = []

        # Set the layer for vision, so that it can be drawn on top of the maze with transparency to show ant vision
        self.vision_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

    def initialize_maze_values(self):
        """Initialize maze with base values and walls"""
        # Set base values
        self.maze.fill(EMPTY_BASE_VALUE)  # Default value for empty spaces

        # Add walls around the map so the ants don't escape
        self.maze[0, :] = WALL_VALUE  # Top wall
        self.maze[-1, :] = WALL_VALUE  # Bottom wall
        self.maze[:, 0] = WALL_VALUE  # Left wall
        self.maze[:, -1] = WALL_VALUE  # Right wall

        # Add random walls
        self.add_random_walls(density=0.2)

        # Add some variation to empty spaces
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.maze[y, x] != WALL_VALUE:  # If not a wall
                    # Add small random variations
                    self.maze[y, x] = EMPTY_BASE_VALUE + random.uniform(-EMPTY_VARIATION, EMPTY_VARIATION)

    def add_random_walls(self, density: float = 0.2):
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if np.random.random() < density:
                    if (x % 2 == 0 and y % 2 == 0 and
                        x + 1 < self.width - 1 and y + 1 < self.height - 1):
                        self.maze[y:y+2, x:x+2] = WALL_VALUE

    def add_food(self, x: int, y: int):
        if self.maze[y, x] != WALL_VALUE:  # Only add food in non-wall spaces
            self.food_positions.append((x, y))
            # Increase position value near food
            self.maze[max(0, y-1):min(self.height, y+2),
                     max(0, x-1):min(self.width, x+2)] += FOOD_INFLUENCE

    def update_vision(self):
        self.vision_surface.fill((0, 0, 0, 0))

        for agent in self.agents:
            surroundings = self.get_agent_surroundings(agent)
            vision_range = 2

            for (dx, dy), value in surroundings.items():
                x = int(agent.visual_x) + dx
                y = int(agent.visual_y) + dy

                if (0 <= x < self.width and 0 <= y < self.height and
                    self.has_line_of_sight(int(agent.visual_x), int(agent.visual_y), x, y)):
                    # Adjust alpha based on distance from agent
                    distance = abs(dx) + abs(dy)
                    alpha = int(64 * (1 - distance / (vision_range * 2)))
                    pygame.draw.rect(
                        self.vision_surface,
                        (0, 255, 0, alpha),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )

    def draw(self):
        self.screen.fill(Colors.WHITE)

        # Draw maze with grayscale values
        for y in range(self.height):
            for x in range(self.width):
                value = self.maze[y, x]
                if value == WALL_VALUE:  # Wall
                    color = Colors.BLACK
                else:  # Empty space with value
                    # Clamp the value between 0 and 1 before converting to intensity
                    clamped_value = max(0.0, min(1.0, value))
                    intensity = int((1 - clamped_value) * 255)
                    # Ensure intensity is within valid range
                    intensity = max(0, min(255, intensity))
                    color = (intensity, intensity, intensity)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        # Draw food
        for fx, fy in self.food_positions:
            pygame.draw.rect(
                self.screen,
                Colors.PURPLE,
                (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size)
            )

        # Draw vision areas
        self.update_vision()
        self.screen.blit(self.vision_surface, (0, 0))

        # Draw agents as circles with their unique colors
        for agent in self.agents:
            pygame.draw.circle(
                self.screen,
                agent.color,
                (int(agent.visual_x * self.cell_size + self.cell_size/2),
                 int(agent.visual_y * self.cell_size + self.cell_size/2)),
                agent.radius
            )

        pygame.display.flip()

    def spawn_agent(self, x: int, y: int):
        """Spawn a new agent at specified position"""
        if self.maze[y, x] != WALL_VALUE:  # Only spawn in non-wall spaces
            agent = Agent(x, y)
            self.agents.append(agent)

    def spawn_random_agents(self, num_agents: int):
        """Spawn multiple agents at random valid positions"""
        empty_cells = [(x, y) for y in range(self.height)
                      for x in range(self.width) if self.maze[y, x] != WALL_VALUE]

        if empty_cells:
            positions = random.sample(empty_cells, min(num_agents, len(empty_cells)))
            for x, y in positions:
                self.spawn_agent(x, y)

    def add_random_food(self, num_food: int):
        """Add food at random valid positions"""
        empty_cells = [(x, y) for y in range(self.height)
                      for x in range(self.width) if self.maze[y, x] != WALL_VALUE]

        if empty_cells:
            positions = random.sample(empty_cells, min(num_food, len(empty_cells)))
            for x, y in positions:
                self.add_food(x, y)

    def has_line_of_sight(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a clear line of sight between two points"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while n > 0:
            if 0 <= y < self.height and 0 <= x < self.width:
                if self.maze[y, x] == WALL_VALUE:  # Wall blocks vision
                    return False
            else:
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

            n -= 1

        return True

    def get_agent_surroundings(self, agent) -> Dict[Tuple[int, int], float]:
        """
        Get the values of the cells in the surroundings of the agent
        """
        surroundings = {}
        vision_range = 2

        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                new_x = agent.x + dx
                new_y = agent.y + dy

                if (0 <= new_x < self.width and
                    0 <= new_y < self.height):
                    if self.has_line_of_sight(agent.x, agent.y, new_x, new_y):
                        surroundings[(dx, dy)] = self.maze[new_y, new_x]
                    else:
                        surroundings[(dx, dy)] = -1
                else:
                    surroundings[(dx, dy)] = -1

        return surroundings

    def update(self):
        """Update all agents"""
        for agent in self.agents:
            maze_info = {
                'surroundings': self.get_agent_surroundings(agent),
            }
            agent.update(maze_info)
