from maze import MazeEnvironment, MazeConfig

def main():
    # Create configuration
    config = MazeConfig(
        width=50,
        height=50,
        cell_size=8,
        num_agents=20,
        initial_food_amount=100.0,
        pheromone_evaporation_rate=0.1,
        pheromone_diffusion_rate=0.05,
        wall_density=0.2
    )

    # Create and run environment
    env = MazeEnvironment(config)
    env.run()

if __name__ == "__main__":
    main()
