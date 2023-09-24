import haiku as hk
import jax.numpy as jnp
import jax
import optax
import random
import numpy as np
import chex
import tree
from typing import Callable
import environment
from environment import Maze, Status
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import jaxlib
from os.path import exists
from abstracts import LearnerState, Transition, ReplayByuffer, run_dqn_episode


# Haiku Neural network function returning a Neural layers

def duelingDeepMazeNetwork(x:  chex.Array):
    out = hk.Linear(64)(x)
    out = jax.nn.relu(out)
    out = hk.Linear(64)(out)
    out = jax.nn.relu(out)
    
    # value function V(x)
    V = hk.Linear(1)(out)
    
    # advantage function A(x)
    A: chex.Array= hk.Linear(4)(out)
    Q = V + (A - A.mean(axis=-1, keepdims=True))
    return Q

# transform the model

# hk.transform Transforms the function having haiku mode to two pure functions init and apply
# init: gives the parameter model.
# apply: applies the change on the method.

# removes the random generator parameter from the apply() function.
dqn_network = hk.without_apply_rng(hk.transform(duelingDeepMazeNetwork))

    
# ----------------------------------------------------------------

class DQNAgent:
    def __init__(self, 
                 init_params_function: Callable[[jax.random.KeyArray], hk.Params],  # a callable function taking jax.random.KeyArray input and returning hk.Params instance output.
                 network_apply_function: Callable[[hk.Params, chex.Array], chex.Array],
                 optimizer: optax.GradientTransformation,
                 gamma: float,
                 epsilon: float,
                 num_actions: int,
                 buffer_capacity: int,
                 batch_size: int,
                 # target_ema: Exponential moving average
                 # method of updating a target neural network slowly over time to make training more stable
                 target_ema: float,
                 seed: int=0,
                 model_name: str="Dueling_DQN_Network_params.pkl",
                 size: int = 20,
                 ) -> None:
        self._gamma = gamma
        self._e = epsilon
        self._target_ema = target_ema
        self._batch_size = batch_size
        self._num_action = num_actions
        
        self._network_apply_func = network_apply_function
        self._optimizer = optimizer
        
        self._buffer= ReplayByuffer(buffer_capacity)
        
        self._state = None
        self.size= size
        
        # initializing the parameters
        params= init_params_function(jax.random.PRNGKey(seed))
        if exists(f"{self.size}/{model_name}"):
            params = self.load_model(params = params, model_name=f"{self.size}/{model_name}")
        
        self._learner_state = LearnerState(
            online_params= params,
            target_params= jax.tree_map(jnp.copy, params), # copying all the parameters w and b from params using jnp.copy func
            opt_state= self._optimizer.init(params), # setting the optimizer
        )
        
        # Jitting the _update function.
        self._update = jax.jit(self._update_function)
    
    def dump_model(self, params: hk.Params,model_name="Dueling_DQN_Network_params.pkl"):
        serialized_params = hk.data_structures.to_state_dict(params)
        with open(f"{self.size}/{model_name}", 'wb') as f:
            pickle.dump(serialized_params, f)
    
    def load_model(self, params: hk.Params, model_name="Dueling_DQN_Network_params.pkl") -> hk.Params:
         with open(model_name, "rb") as f:
            loaded_params = pickle.load(f)
            params = hk.data_structures.merge(params, loaded_params)
         return params
    
    def _update_function(
                self,
                state: LearnerState,
                batch: Transition,
                ) -> tuple[LearnerState, chex.Array]:
        # takes the Learner state and batch of transitions as a data and apply the update
        # and returns the [updated learner state] and [loss] together as a tuple.
        
        # The transitions we are passing to the functions is a list of Transition but the type of the parameter is just a Transition 
        # As we are jitting this function when calling the whenever we call this function with same learner state and list of states
        # it will iterate over the transition and same state instance to give us the results.
        loss, gradient = jax.value_and_grad(self._loss_function)(
            state.online_params,
            target_params = state.target_params,
            transitions=batch
        )
        
        
        # Apply gradient based on the loss function.
        updates, opt_state = self._optimizer.update(gradient, state.opt_state)
        online_params = optax.apply_updates(state.online_params, updates)
        
        ema = self._target_ema
        target_params = jax.tree_map(
            lambda online, target: ema * target   + (1-ema)* online,
            online_params,
            state.target_params,
        )
        
        # return the updates Learner state and loss.
        return LearnerState(
            online_params= online_params,
            target_params= target_params,
            opt_state= opt_state,
        ), loss

    # being in the given state, select an action
    def select_action(self, 
                      state: chex.Array, 
                      eval: bool) -> chex.Array:
        """Picks the next action using an epsilon greedy policy.

        Args:
        obersation: observed state of the environment.
        eval: if True the agent is acting in evaluation mode (which means it only
            acts according to the best policy it knows.)
        """
        # if not in eval or <= epsilon
        # Exploitation
        if eval or np.random.uniform() > self._e:
            # A function in the JAX library that adds 
            # a new axis (dimension) to a JAX NumPy array. 
            # It allows you to change the shape of the array 
            # by increasing its dimensionality.
            # We do this because the network expects a batch dimention so we add one here.
            state_expand = jnp.expand_dims(state, axis=0)
            
            # Greedy action selection
            q_values = self._network_apply_func(
                self._learner_state.online_params,
                state_expand,
            )
            q_values = jnp.squeeze(q_values, axis=0)
            val = jnp.argmax(q_values, axis=-1)
            # Check if the maximum index is in the list
            if isinstance(val, list) or isinstance(val, jaxlib.xla_extension.ArrayImpl):
                return int(val[0])
            # If it's not in the list, select a default value
            return int(val)
        
        # Exploration
        return np.random.randint(self._num_action)

    def first_state(self, state: chex.Array):
        self._state = state
    
    def _loss_function(self, 
                       online_params: hk.Params,
                       target_params: hk.Params,
                       transitions: Transition
                       ) -> chex.Array: # returns a loss
        
        next_qs = jnp.argmax(self._network_apply_func(online_params,transitions.next_state), axis=-1, keepdims=True)

        # Step 2: Calculate masked_next_qs
        masked_next_qs = jnp.take_along_axis(self._network_apply_func(target_params, transitions.next_state), next_qs, axis=-1).squeeze()
        masked_next_qs = jnp.squeeze(masked_next_qs)
        # target_q_actions = self._network_apply_func(target_params, transitions.next_state)
        
        # If the state is WIN or LOSE the transition.done will be true.
        y = jnp.where(
            transitions.done,
            transitions.reward,
            # transitions.reward + self._gamma * jnp.max(masked_next_qs, axis=-1)
            transitions.reward + self._gamma * masked_next_qs
        )
        
        online_q_values = self._network_apply_func(online_params, transitions.state) 
        
        online_q_value_at_action = jnp.take_along_axis(
            online_q_values,
            transitions.action[:, None],
            axis=-1,
        )
        online_q_value_at_action = jnp.squeeze(
          online_q_value_at_action,
          axis=-1,
        )
        loss = 0.5 * jnp.mean(jnp.square(y - online_q_value_at_action))
        return loss

    def observe(self, state, action, next_state, reward, status):
        observation = Transition(
            state= state,
            action= action,
            reward= reward,
            done= status in (Status.WIN, Status.LOSE),
            next_state= next_state,
        )
        self._buffer.add(observation)

        # TODO: what is the use of current observation.
        # I don't know, yet.
        # Setting the current observation
        self._state = observation.state
    
    def update(self) -> chex.Array | None:
        
        if self._buffer.size> self._batch_size:
            # apply the update using the jitted _update function.
            batch = self._buffer.sample_batch(self._batch_size)
            self._learner_state, loss = self._update(
                self._learner_state, 
                batch)
            return loss
        return None

default_env = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])


# default_env = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0]
# ])

# default_env = np.array([
#     [0, 1, 0, 0],
#     [0, 1, 0, 1],
#     [0, 0, 0, 1],
#     [0, 1, 0, 0]
# ])


class DuelingDQNNetworkModel:
    """ Prediction model which uses Q-learning and a neural network which replays past moves.

        The network learns by replaying a batch of training moves. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    def __init__(self, 
                 environment: Maze, 
                 agent: DQNAgent, 
                 num_episodes: int = 300000,
                 batch_size=64,
                 evaluation_interval: int=500,
                 ):
        super().__init__()
        self._env = environment
        self._agent = agent
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        
        # Number of evaluation episodes when evaluating.
        
        self.all_rewards = []
        self.episodes_logs = []
        self.all_rewards = []
        self.all_losses = []
        self.online_test_q_values = []
        self.target_test_q_values = []
        
        self.win_rate_history= []
        
        # Logging the q values 
        self.num_eval_episodes = 10
        
    def predict(self, state) -> int:
        if isinstance(state, tuple):
            state= jnp.asarray(state)
        return self._agent.select_action(state=state,eval=True)

    def train(self):
        print(f"Dueling DQN training")
        print(f"Episode number:\t| Average reward on {self.num_eval_episodes} eval episodes")
        print("------------------------------------------------------")
        timestamp = datetime.now()
        timelapse = None
        for episode in range(self.num_episodes):
            run_dqn_episode(
                dqn_agent= self._agent,
                env=self._env,
                eval=False,
            )
            loss = self._agent.update()
            
            # Log the loss for later plotting.
            self.all_losses.append(loss)
            
            # TODO: testing actions logging for later use.
            
            if episode % self.evaluation_interval == 0:
                reward = np.mean([
                    run_dqn_episode(self._agent, self._env, eval=True)
                    for _ in range(self.num_eval_episodes)
                ])
                # Print how much reward the agent accumulated on average.
                print(f"\t{episode}\t|\t{reward}")
                self.all_rewards.append(reward)
                self.episodes_logs.append(episode)
                
                # -------------------------------------------------------
                
                # Evaluate by running in all cells.
                w_all, win_rate = self._env.check_win_all(self, model_type=environment.maze.ModelType.DQN)
                self.win_rate_history.append((episode, win_rate))
                
                # This will stop at convergence.
                if w_all:
                    print("Won from all start cells, stop learning\n")
                    # timelapse = datetime.now() - timestamp
                    break
        timelapse = (datetime.now() - timestamp).total_seconds()
        print("Time taken to finish: ", timelapse, " seconds")
        
        self.dump_model(self._agent._learner_state.target_params)
        


    def dump_model(self, params: hk.Params,model_name="Dueling_DQN_Network_params.pkl"):
        with open(f"{self._env.maze.shape[0]}/{model_name}", 'wb') as f:
            pickle.dump(params, f)
    
     
    # draw the training loss.
    def draw(self):
        self._env.draw()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].set_xlabel('Number of training episodes')
        axs[0].set_ylabel('Average return')
        axs[0].plot(self.episodes_logs, self.all_rewards)
        axs[0].set_ylim([-1.1, 1.1])
        axs[1].set_xlabel('Number of updates')
        axs[1].set_ylabel('Average loss')
        axs[1].plot(np.asarray(self.all_losses).T)
        axs[1].set_ylim([0, 0.25]);   
    
    
    
    
    
    
# ----------------------------------------------------------------


# Create the neural network pure functions.
dqn_network = hk.without_apply_rng(hk.transform(duelingDeepMazeNetwork))

# Bind a dummy observation to the init function so the agent doesn't have to.
dummy_observation = jnp.zeros((2,), float)
init_params_fn = lambda rng: dqn_network.init(rng, dummy_observation[None, ...])

env = Maze(default_env)
model = DuelingDQNNetworkModel(
    environment= env,
    agent=DQNAgent(
        init_params_function=init_params_fn ,
        network_apply_function= dqn_network.apply,
        optimizer= optax.adam(learning_rate=3e-4),
        gamma=0.9,
        epsilon=0.3,
        num_actions= len(env.actions),
        buffer_capacity= 10000,
        batch_size= 256,
        target_ema= .89,
        size=env.maze.shape[0],
    ),
)


if __name__=='__main__':
    model.train()

# Evaluation hyperparameters
