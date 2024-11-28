import pygame
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random
from colors import Colors
from agent import FoodSource, PheromoneLayer, Agent

@dataclass
class MazeConfig:
    width: int = 100
    height: int = 100
    cell_size: int = 8
    num_agents: int = 20
    initial_food_amount: float = 100.0
    pheromone_evaporation_rate: float = 0.1
    pheromone_diffusion_rate: float = 0.05
    food_deposit_amount: float = 1.0
    wall_density: float = 0.2

class MazeEnvironment:
    def __init__(self, config: MazeConfig):
        pygame.init()
        self.config = config

        # Initialize display
        self.screen_width = config.width * config.cell_size
        self.screen_height = config.height * config.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ant Colony Optimization")

        # Define nest position before initializing maze
        self.nest_position = (2, 2)

        # Initialize layers
        self.maze = np.zeros((config.height, config.width), dtype=float)
        self.initialize_maze()

        # Create pheromone layer
        self.pheromone_layer = PheromoneLayer(
            width=config.width,
            height=config.height,
            evaporation_rate=config.pheromone_evaporation_rate,
            diffusion_rate=config.pheromone_diffusion_rate
        )

        # Initialize game elements
        self.agents: List[Agent] = []
        self.food_sources: List[FoodSource] = []

        # Visualization layers
        self.vision_surface = pygame.Surface(
            (self.screen_width, self.screen_height),
            pygame.SRCALPHA
        )

        # Initialize game state
        self.spawn_initial_food()
        self.spawn_agents()

    def initialize_maze(self):
        """Initialize maze with improved wall generation"""
        # Set base values for empty spaces
        self.maze.fill(0.5)

        # Add border walls
        self.maze[0, :] = 1.0  # Top wall
        self.maze[-1, :] = 1.0  # Bottom wall
        self.maze[:, 0] = 1.0  # Left wall
        self.maze[:, -1] = 1.0  # Right wall

        # Add organic-looking walls using noise
        noise = np.random.rand(self.config.height, self.config.width)
        smoothed_noise = np.zeros_like(noise)

        # Simple smoothing
        for y in range(1, self.config.height-1):
            for x in range(1, self.config.width-1):
                smoothed_noise[y, x] = np.mean(noise[y-1:y+2, x-1:x+2])

        # Create walls where smoothed noise exceeds threshold
        wall_threshold = 1.0 - self.config.wall_density
        wall_mask = smoothed_noise > wall_threshold
        self.maze[wall_mask] = 1.0

        # Ensure nest area is clear
        nest_x, nest_y = self.nest_position
        self.maze[nest_y-2:nest_y+3, nest_x-2:nest_x+3] = 0.3  # Mark nest area

        # Ensure paths are connected (flood fill from nest)
        self._ensure_connected_paths()

    def _ensure_connected_paths(self):
        """Ensure all spaces are reachable from nest using flood fill"""
        visited = np.zeros_like(self.maze, dtype=bool)
        stack: List[Tuple[int, int]] = [self.nest_position]

        while stack:
            x, y = stack.pop()
            if not visited[y, x]:
                visited[y, x] = True

                # Check neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < self.config.width and
                        0 <= new_y < self.config.height and
                        not visited[new_y, new_x] and
                        self.maze[new_y, new_x] != 1.0):
                        stack.append((new_x, new_y))

        # Create paths where needed
        for y in range(1, self.config.height-1):
            for x in range(1, self.config.width-1):
                if not visited[y, x] and self.maze[y, x] != 1.0:
                    # Find nearest visited cell and create path
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if visited[y+dy, x+dx]:
                                self.maze[y, x] = 0.5
                                break

    def add_random_walls(self, density: float):
        """Add random walls to the maze"""
        for y in range(2, self.config.height-2, 2):
            for x in range(2, self.config.width-2, 2):
                if random.random() < density:
                    self.maze[y:y+2, x:x+2] = 1.0

    def spawn_initial_food(self):
        """Place initial food source"""
        while True:
            x = random.randint(10, self.config.width-10)
            y = random.randint(10, self.config.height-10)

            # Check if position is valid (not wall or near nest)
            if (self.maze[y, x] != 1.0 and
                abs(x - self.nest_position[0]) > 5 and
                abs(y - self.nest_position[1]) > 5):

                self.food_sources.append(FoodSource(
                    x=x,
                    y=y,
                    amount=self.config.initial_food_amount
                ))
                break

    def spawn_agents(self):
        """Spawn initial agents near nest"""
        nest_x, nest_y = self.nest_position

        for _ in range(self.config.num_agents):
            # Spawn in area around nest
            while True:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                x = nest_x + dx
                y = nest_y + dy

                if 0 <= x < self.config.width and 0 <= y < self.config.height:
                    if self.maze[y, x] != 1.0:  # Not a wall
                        agent = Agent(x, y, self.config.width, self.config.height)
                        self.agents.append(agent)
                        break

    def get_agent_surroundings(self, agent: Agent) -> Dict:
        """Get information about agent's surroundings"""
        surroundings = {}
        pheromones = {}

        # Check surrounding cells
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                new_x = agent.x + dx
                new_y = agent.y + dy

                if 0 <= new_x < self.config.width and 0 <= new_y < self.config.height:
                    surroundings[(dx, dy)] = self.maze[new_y, new_x]
                    pheromones[(dx, dy)] = self.pheromone_layer.get_pheromone(new_x, new_y)
                else:
                    surroundings[(dx, dy)] = 1.0  # Treat out of bounds as wall
                    pheromones[(dx, dy)] = 0.0

        # Check for food at current position
        food_here = False
        food_amount = 0.0
        for food in self.food_sources:
            if food.x == agent.x and food.y == agent.y:
                food_here = True
                food_amount = food.amount
                break

        return {
            'surroundings': surroundings,
            'pheromones': pheromones,
            'food_here': food_here,
            'food_amount': food_amount
        }

    def update(self):
        """Update game state"""
        # Update pheromone layer
        self.pheromone_layer.update()

        # Update agents
        for agent in self.agents:
            # Get surroundings information
            surroundings_info = self.get_agent_surroundings(agent)

            # Update agent
            prev_x, prev_y = agent.x, agent.y
            agent.update(surroundings_info)

            # Handle pheromone deposition
            if agent.carrying_food:
                # Deposit pheromone on path when carrying food
                self.pheromone_layer.add_pheromone(
                    prev_x, prev_y,
                    agent.food_amount * 0.1  # Pheromone amount proportional to food
                )

            # Handle food collection
            if agent.carrying_food and (agent.x, agent.y) == self.nest_position:
                food_amount = agent.deposit_food()
                # Could track total food collected here

        # Update food sources - remove depleted ones
        self.food_sources = [food for food in self.food_sources if not food.is_depleted]

    def draw(self):
        """Render the game state"""
        self.screen.fill(Colors.WHITE)

        # Draw maze
        for y in range(self.config.height):
            for x in range(self.config.width):
                value = self.maze[y, x]
                if value == 1.0:  # Wall
                    color = Colors.BLACK
                else:  # Empty space or nest
                    # Mix base color with pheromone intensity
                    pheromone = self.pheromone_layer.get_pheromone(x, y)
                    base = max(0, min(255, int((1 - value) * 255)))  # Clamp base value
                    blue = max(0, min(255, int(pheromone * 255)))  # Clamp blue value
                    color = (base, base, blue)

                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * self.config.cell_size,
                     y * self.config.cell_size,
                     self.config.cell_size,
                     self.config.cell_size)
                )

        # Draw food sources
        for food in self.food_sources:
            # Color intensity based on remaining food
            intensity = max(0, min(255, int((food.amount / self.config.initial_food_amount) * 255)))
            color = (intensity, 0, intensity)  # Purple-ish
            pygame.draw.rect(
                self.screen,
                color,
                (food.x * self.config.cell_size,
                 food.y * self.config.cell_size,
                 self.config.cell_size,
                 self.config.cell_size)
            )

        # Draw agents
        for agent in self.agents:
            color = Colors.RED if agent.carrying_food else Colors.BLUE
            pygame.draw.circle(
                self.screen,
                color,
                (int(agent.x * self.config.cell_size + self.config.cell_size/2),
                 int(agent.y * self.config.cell_size + self.config.cell_size/2)),
                3  # radius
            )

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update()
            self.draw()
            clock.tick(60)

        pygame.quit()
