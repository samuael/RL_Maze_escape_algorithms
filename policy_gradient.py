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

def policyGradientMazeNetwork(x:  chex.Array):
    out = hk.Linear(64)(x)
    out = jax.nn.relu(out)
    out = hk.Linear(64)(out)
    out = jax.nn.relu(out)
    return hk.Linear(4)(out)

# transform the model

# removes the random generator parameter from the apply() function.
dqn_network = hk.without_apply_rng(hk.transform(policyGradientMazeNetwork))


# ----------------------------------------------------------------

class PolicyGradientAgent:
    def __init__(self, init_params_function: Callable[[jax.random.KeyArray], hk.Params],  # a callable function taking jax.random.KeyArray input and returning hk.Params instance output.
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
                 model_name: str="Policy_gradient_network_params.pkl",
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
            opt_state= self._optimizer.init(params), # setting the optimizer
            target_params= jax.tree_map(jnp.copy, params),
        )

        # Jitting the _update function.
        self._update = jax.jit(self._update_function)

    def dump_model(self, params: hk.Params,model_name="policy_gradient_params.pkl"):
        serialized_params = hk.data_structures.to_state_dict(params)
        with open(f"{self.size}/{model_name}", 'wb') as f:
            pickle.dump(serialized_params, f)

    def load_model(self, params: hk.Params, model_name="policy_gradient_params.pkl") -> hk.Params:
         with open(model_name, "rb") as f:
            loaded_params = pickle.load(f)
            params = hk.data_structures.merge(params, loaded_params)
         return params
    def softmax_cross_entropy_loss(self, predicted_logits, true_labels):
        # Compute softmax
        softmax_logits = jax.nn.softmax(predicted_logits, axis=-1)

        # Compute log probabilities
        log_probs = jnp.log(softmax_logits)

        # Gather the log probabilities for the true labels
        true_label_log_probs = jnp.sum(log_probs * jax.nn.one_hot(true_labels, predicted_logits.shape[-1]), axis=-1)

        # Compute the negative log-likelihood (cross-entropy loss)
        loss = -true_label_log_probs
        return jnp.mean(loss)

    def _loss_function(self, params, states, actions, cum_rewards):
        states = jnp.asarray(states).squeeze()
        actions = jnp.asarray(actions)
        logits = self.apply_func(params, states)
        log_probs = self.softmax_cross_entropy_loss(logits, actions)
        loss = jnp.mean(-jnp.asarray(log_probs) * cum_rewards, axis=-1)
        return loss

    def _update_function(
                self,
                state: LearnerState,
                batch: Transition) -> tuple[LearnerState, chex.Array]:
        loss, grads = jax.value_and_grad(self._loss_function)(state.online_params, batch)
        updates, opt_state = self._optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.online_params, updates)

        # return the updates Learner state and loss.
        return LearnerState(
            online_params= params,
            opt_state= opt_state,
            target_params=None,
        ), loss

    # being in the given state, select an action
    def select_action(self, state: chex.Array, eval: bool) -> chex.Array:
        # Exploitation
        if eval or np.random.uniform() > self._e:
            state_expand = jnp.expand_dims(state, axis=0)

            # Greedy action selection
            p_values = self._network_apply_func(
                self._learner_state.online_params,
                state_expand,
            )
            p_values = jnp.squeeze(p_values, axis=0)
            val = jnp.argmax(p_values, axis=-1)
            # Check if the maximum index is in the list
            if isinstance(val, list) or isinstance(val, jaxlib.xla_extension.ArrayImpl):
                return int(val[0])
            # If it's not in the list, select a default value
            return int(val)

        # Exploration
        return np.random.randint(self._num_action)

    def first_state(self, state: chex.Array):
        self._state = state

    def buffer_size(self)->int:
        return self._buffer.size, self._batch_size

    def _loss_function(self,
                       param: hk.Params,
                       transition: Transition
                       ) -> chex.Array: # returns a loss

        states = jnp.asarray(transition.state).squeeze()
        actions = jnp.asarray(transition.action)
        logits = self._network_apply_func(param, states)
        log_probs = self.softmax_cross_entropy_loss(logits, actions)
        loss = jnp.mean(-jnp.asarray(log_probs) * transition.reward, axis=-1)
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

        # if self._buffer.size> self._batch_size:
        # apply the update using the jitted _update function.
        batch = self._buffer.sample_batch(self._batch_size)
        self._learner_state, loss = self._update(
            self._learner_state, batch)
        return loss
        # return None

default_env = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
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

default_env = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
])


class PolicyGradientModel:
    """ Prediction model which uses Q-learning and a neural network which 
replays past moves.

        The network learns by replaying a batch of training moves. The 
training algorithm ensures that
        the game is started from every possible cell. Training ends after 
a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win 
rate).
    """
    def __init__(self,
                 environment: Maze,
                 agent: PolicyGradientAgent,
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
        print(f"Policy Gradient training")
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
                w_all, win_rate = self._env.check_win_all(self, model_type=environment.maze.ModelType.AgentBased)
                self.win_rate_history.append((episode, win_rate))

                # This will stop at convergence.
                if w_all:
                    print("Won from all start cells, stop learning\n")
                    # timelapse = datetime.now() - timestamp
                    break
        timelapse = (datetime.now() - timestamp).total_seconds()
        print("Time taken to finish: ", timelapse, " seconds")

        self.dump_model(self._agent._learner_state.target_params)



    def dump_model(self, params: hk.Params,model_name="policy_gradient_params.pkl"):
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
pg_network = hk.without_apply_rng(hk.transform(policyGradientMazeNetwork))

# Bind a dummy observation to the init function so the agent doesn't have to.
dummy_observation = jnp.zeros((2,), float)
init_params_fn = lambda rng: pg_network.init(rng, dummy_observation[None, 
...])

env = Maze(default_env)
model = PolicyGradientModel(
    environment= env,
    agent=PolicyGradientAgent(
        init_params_function=init_params_fn ,
        network_apply_function= pg_network.apply,
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

