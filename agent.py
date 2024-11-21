import random
from typing import List, Tuple, Dict
import numpy as np
from colors import ANT_COLORS

class Agent:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.move_cooldown = 0
        self.move_delay = 5
        self.visual_x = x
        self.visual_y = y
        self.color = random.choice(ANT_COLORS)
        self.radius = 3

    def move_randomly(self, valid_moves) -> bool:
        """
        Make a random move from the valid moves provided by the environment
        """
        if self.move_cooldown > 0:
            return False

        if valid_moves:
            new_x, new_y, _ = random.choice(valid_moves)
            self.x = new_x
            self.y = new_y
            self.move_cooldown = self.move_delay
            return True

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
            valid_moves = maze_info['valid_moves']
            self.move_randomly(valid_moves)
        self.update_visual_position()
