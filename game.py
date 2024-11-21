import pygame
from maze import MazeEnvironment

def main():
    """
    Main function with the game loop
    """
    # Maze 100x100 with 8x8 pixel cells
    env = MazeEnvironment(100, 100, 8)

    env.add_random_food(30)
    env.spawn_random_agents(50)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.update()
        env.draw()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
