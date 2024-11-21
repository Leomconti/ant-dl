import random
from colors import ANT_COLORS

class Agent:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.move_cooldown = 0
        self.move_delay = 60 # Delay para mexer, 60 = 1 segundp
        self.visual_x = x
        self.visual_y = y
        self.color = random.choice(ANT_COLORS)
        self.radius = 3
        self.move = random.choice([self.move_randomly, self.move_with_food])

    def move_randomly(self, surroundings) -> bool:
        print("move_randomly")
        """
        Make a random move based on the surroundings matrix
        """
        if self.move_cooldown > 0:
            return False

        # Get possible moves (adjacent cells)
        possible_moves = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
            if surroundings.get((dx, dy), -1) != -1:  # If cell is reachable
                possible_moves.append((self.x + dx, self.y + dy))

        # Here is the possible moves
        if possible_moves:
            new_x, new_y = random.choice(possible_moves)
            self.x = new_x
            self.y = new_y
            self.move_cooldown = self.move_delay
            return True

        return False

    def move_with_food(self, surroundings) -> bool:
        print("move_with_food")
        """
        Make a move towards the food if there is any in the surroundings
        """
        return False

    def update_visual_position(self):
        move_speed = 0.2
        dx = self.x - self.visual_x
        dy = self.y - self.visual_y
        self.visual_x += dx * move_speed
        self.visual_y += dy * move_speed

    def update(self, maze_info):
        """
        Update agent based only on information provided by the environment
        """
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        else:
            surroundings = maze_info['surroundings']
            self.move(surroundings)
        self.update_visual_position()
