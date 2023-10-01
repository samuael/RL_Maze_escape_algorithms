### Reinforcement Learning Methods for the Maze Game

##### Solving the problem using DQN, DDQN, DDQN, and Policy gradient

The table below gives an impression of the relative performance of each of these models :

Result recorded on 10x10 grid of maze by running 10x for these three(3) algorithms.


| Model | Trained | Average no of episodes | Average training time |
| --- | --- | --- | --- | 
| DQN | 10 times | 9700 | 504.623 sec |
| DDQN  | 10 times | 12700 | 357.61 sec |
| SARSA  | 10 times |   300    |   43.31 sec |
| A2C | 10 times | 8200 | 510.7 sec |

Implemented but have a bad result.

|Policy grandient|


Required libraries:

- matplotlib, numpy,Jax, optax, haiku, pickle, chex, and jaxlib

![](Untitled.gif)


- To run the pre-trained model run `python main.py --`