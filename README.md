# Machine Learning - Ant Colony

- Define values for all things:
  -1 = can't see
  0 = path
  1 = wall
  2 = food
  3 = home

---

Regras para rodar:

- agentes spawna aleatorio pelo mapa
- o ninho define X,Y antes de dar play - SABE ONDE ESTA O NINHO, 2,2 - sabe x,y proprio
- as comidas define X,Y antes de dar play - a comida tem uma quantidade de comida - 100-200
- comida tem valor so no seu bloco

- pode clicar no meio do jogo botando a comida onde quiser

---

Create ML algorithm where the ants learn how to navigate the maze, to get food and return it to the nest.

1st do hard coded dumb algorithm
Later to genetic
Then do reinforcement learning

# TODO:

- I want to implement ant colony optimization algorithm, basically the ants will learn how to navigate the maze, to get food and return it to the nest.
- A food block should have a value, which is the amount of food it has. When the ant eats the food, the food block value decreases.
- Each agent should have memory, implement this.
- implement pheromone, when the ant eats the food, it will leave a pheromone trail back to the nest ( only on the way back, to avoid recursion)
- every ant should know where the nest is, so that they can return to it
- implement pheromone influence on the ants decision making, which means pheromone has a value, and should increase as more ants pass through the path

# Constraints:

- start with just 1 food block with a value of 100 in a place you can select ( ants should not know of course!)
- start with 20 ants
- maze is 100x100 with 8x8 pixel cells, or something like that

# Ideas:

- First think what are some ways we can do that
- Then think how we can implement, what do we need to change in the core of the application, on the maze and game, to be able to iterate with different algorithms on the agents themselves
- Maybe use state machine for the ants, so that they have different states and behaviors, like state = going_to_food, state = looking_for_food, state = carrying_food_to_nest
