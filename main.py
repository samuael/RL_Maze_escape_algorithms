from environment.maze import Maze, Render, ModelType
import numpy as np
import optax
import os
from Dueling_DQN import DuelingDQNNetworkModel, DQNAgent, duelingDeepMazeNetwork
import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt

default_env = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
])

game = Maze(default_env)


dqn_network = hk.without_apply_rng(hk.transform(duelingDeepMazeNetwork))

# Bind a dummy observation to the init function so the agent doesn't have to.
dummy_observation = jnp.zeros((2,), float)
init_params_fn = lambda rng: dqn_network.init(rng, dummy_observation[None, ...])

model = DuelingDQNNetworkModel(
    environment= game,
    agent=DQNAgent(
        init_params_function=init_params_fn ,
        network_apply_function= dqn_network.apply,
        optimizer= optax.adam(learning_rate=3e-4),
        gamma=0.92,
        epsilon=0.1,
        num_actions= len(game.actions),
        buffer_capacity= 10000,
        batch_size= 256,
        target_ema= .8,
        size=game.maze.shape[0],
    ),
)
# print(len(default_env))
game.render(Render.MOVES, timeout=120000)
for x in range(10):
    for y in range(10):
        try:
            game.play(model, start_cell=(x, y), model_type=ModelType.AgentBased)
            # plt.figure().show
            # plt.show()
        except:
            # continue
            pass