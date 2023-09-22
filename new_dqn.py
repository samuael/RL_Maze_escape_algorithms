#@title Imports  { form-width: "30%" }


# from IPython.display import clear_output
# clear_output()

import bsuite.environments.catch as dm_catch
import chex
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tree
from typing import Callable

# Filter out warnings as they are distracting.
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=3, suppress=1)

# %matplotlib inline

# @title Interacting with the environment.

environment = dm_catch.Catch(rows=5)

# ----------------------------------------------------------------

#@title **[Implement]** Uniform Replay Buffer { form-width: "30%" }

@chex.dataclass
class Transition:
  observation: chex.Array
  action: chex.Array
  reward: chex.Array
  done: chex.Array
  next_observation: chex.Array


class ReplayBuffer:
  """Fixed-size buffer to store transition tuples."""

  def __init__(self, buffer_capacity: int):
      """Initialize a ReplayBuffer object.
      Args:
          buffer_capacity (int): maximum allowed size of the replay buffer.
      """
      self._memory = list()
      self._maxlen = buffer_capacity

  @property
  def size(self) -> int:
    # Return the current number of elements in the buffer.
    return len(self._memory)

  def add(self, transition: Transition) -> None:
      """Add a new transition to memory."""
      # Your code here !
      if self.size >= self._maxlen:
        self._memory.pop(0)
      self._memory.append(transition)

  def sample_batch(self, batch_size: int) -> Transition:
    """Randomly sample a batch of experiences from memory."""
    assert len(self._memory) >= batch_size, 'Insuficient number of transitions in replay buffer'
    # Your code here !
    transitions: list[Transition] = random.choices(self._memory, k=batch_size)
    return tree.map_structure(lambda *x: np.stack(x), *transitions)

# ----------------------------------------------------------------

# @title Test the replay buffer

# batch_size = 32
# buffer = ReplayBuffer(100)

# for _ in range(batch_size):
#   ts = environment.reset()

#   buffer.add(
#       Transition(
#           observation=ts.observation,
#           action=1,
#           reward=ts.reward or 0.,
#           done=ts.last(),
#           next_observation=ts.observation,
#       )
#   )

# assert buffer.sample_batch(batch_size).observation.shape[0] == batch_size

# ----------------------------------------------------------------

#@title **[Implement]** Catch Network { form-width: "30%" }
num_rows = 5
num_columns = 5

environment = dm_catch.Catch(rows=num_rows, columns=num_columns)
print(environment.action_spec())
num_actions = 3 # environment.action_spec().num_values

def catch_network(x: chex.Array):
  # Your code here !
  out = hk.Flatten()(x)  # For example: [B, 5, 5] -> [B, 25].
  out = hk.Linear(64)(out)
  out = jax.nn.relu(out)
  out = hk.Linear(num_actions)(out)
  return out

# Create the neural network pure functions.
dqn_network = hk.without_apply_rng(hk.transform(catch_network))

# ----------------------------------------------------------------



#@title **[Implement]** DQN agent

@chex.dataclass
class LearnerState:
  """"Container for all variables needed for training."""
  online_params: hk.Params
  target_params: hk.Params
  opt_state: optax.OptState

class DQNAgent:
  """Implementation of the DQN agent."""

  def __init__(
      self,
      init_params_fn: Callable[[jax.random.KeyArray], hk.Params],
      network_apply_fn: Callable[[hk.Params, chex.Array], chex.Array],
      optimizer: optax.GradientTransformation,
      gamma: float,
      epsilon: float,
      num_actions: int,
      buffer_capacity: int,
      batch_size: int,
      target_ema: float,
      seed: int = 0,
  ) -> None:
    """Initializes the DQN agent.

    Args:
      init_params_fn: the pure function which initializes the network parameters.
      network_apply_fn: the pure function corresponding to the desired DQN network.
      optimizer: the optimizer used to minimize the DQN loss.
      gamma: the agent's discount factor.
      epsilon: probability to perform a random exploration when picking a new action.
      num_actions: number of actions in the environment's action space.
      buffer_capacity: capacity of the replay buffer.
      batch_size: batch size when updating the online network.
      target_ema: coefficient for the exponential moving average computation of
        the target network parameters.
      seed: seed of the random generator.
    """
    self._gamma = gamma
    self._epsilon = epsilon
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._target_ema = target_ema

    # Set the neural network and optimizer.
    self._network_apply_fn = network_apply_fn
    self._optimizer = optimizer

    # Initialize the replay buffer.
    self._buffer = ReplayBuffer(buffer_capacity)

    # Always store the current observation so we can create transitions.
    self._observation = None

    # Initialize the network's parameters.
    params = init_params_fn(jax.random.PRNGKey(seed))

    # Initialize the learner state.
    # Your code here !
    self._learner_state = LearnerState(
        online_params=params,
        target_params=jax.tree_map(jnp.copy, params),
        opt_state=self._optimizer.init(params),
    )

    # JIT the update step.
    self._update = jax.jit(self._update_fn)

  def observe_first(self, observation: chex.Array) -> None:
    self._observation = observation

  def select_action(
      self,
      observation: chex.Array,
      eval: bool,
  ) -> chex.Array:
    """Picks the next action using an epsilon greedy policy.

    Args:
      obersation: observed state of the environment.
      eval: if True the agent is acting in evaluation mode (which means it only
        acts according to the best policy it knows.)
    """
    # Fill in this function to act using an epsilon-greedy policy.
    # Your code here !
    if eval or np.random.uniform() > self._epsilon:
      # The network expects a batch dimension so we add one here.
      observation = jnp.expand_dims(observation, axis=0)
      # Greedy action selection.
      q_values = self._network_apply_fn(
          self._learner_state.target_params,
          observation
      )
      # Remove the batch dimension that was added above.
      q_values = jnp.squeeze(q_values, axis=0)
      action = jnp.argmax(q_values, axis=-1)
    else:
      # Random action selection.
      action = np.random.randint(self._num_actions)

    return action

  def _loss_fn(
      self,
      online_params: hk.Params,
      target_params: hk.Params,
      transition: Transition,
  ) -> chex.Array:
      """Computes the Q-learning loss

      Args:
        online_params: parameters of the online network.
        target_params: parameters of the target network.
        transition: container of transition quantities (s, a, r, done, s')
      Returns:
        The Q-learning loss.
      """
      # Your code here !
      target_q_values = self._network_apply_fn(
          target_params,
          transition.next_observation,
      )
      y = jnp.where(
          transition.done,
          transition.reward,
          transition.reward + self._gamma * jnp.max(target_q_values, axis=-1),
      )
      online_q_values = self._network_apply_fn(
          online_params,
          transition.observation,
      )
      online_q_value_at_action = jnp.take_along_axis(
          online_q_values,
          transition.action[:, None],
          axis=-1,
      )
      online_q_value_at_action = jnp.squeeze(
          online_q_value_at_action,
          axis=-1,
      )
      loss = 0.5 * jnp.mean(jnp.square(y - online_q_value_at_action))

      return loss

  def _update_fn(
      self,
      state: LearnerState,
      batch: Transition,
  ) -> tuple[LearnerState, chex.Array]:
    """Get the next learner state given the current batch of transitions.

    Args:
      state: the current learner state.
      batch: batch of transitions (st, at, rt, done_t, stp1)
    Returns:
      A tuple of:
        - the updated learner state, and
        - the loss incurred by the previous learner state given the batch.
    """

    # Compute gradients
    # Your code here !
    loss, grad = jax.value_and_grad(self._loss_fn)(
        state.online_params,
        target_params=state.target_params,
        transition=batch
    )

    # Apply gradients
    # Your code here !
    updates, opt_state = self._optimizer.update(grad, state.opt_state)
    online_params = optax.apply_updates(state.online_params, updates)

    # Update target network params as:
    #   target_params <- ema * target_params + (1 - ema) * online_params
    # You code here !
    ema = self._target_ema
    target_params = jax.tree_map(
        lambda online, target: ema * target + (1 - ema) * online,
        online_params,
        state.target_params,
    )

    new_state = LearnerState(
      online_params=online_params,
      target_params=target_params,
      opt_state=opt_state,
    )

    return new_state, loss

  def observe(self, action: chex.Array, timestep: dm_env.TimeStep) -> None:
    """Updates the agent from the given observations.

    Args:
      action: action performed at time t.
      timestep: timestep returned by the environment after
    """
    # Create the transition.
    transition = Transition(
        # Current observation.
        observation=self._observation,
        # Action taken given that observation.
        action=action,
        # Result of taking the action.
        reward=timestep.reward,
        done=timestep.last(),
        next_observation=timestep.observation,
    )
    # Add the transition to the replay buffer.
    self._buffer.add(transition)
    # Update the current observation.
    self._observation = timestep.observation

  def update(self) -> chex.Array | None:
    """Performs an update step if there is enough transitions in the buffer.
    Returns: DQN loss obtained when updating the online network or None if
      there was not enough data.
    """
    if self._buffer.size >= self._batch_size:
      batch = self._buffer.sample_batch(self._batch_size)
      self._learner_state, loss = self._update(self._learner_state, batch)
      return loss
    return None


# ----------------------------------------------------------------

# @title Define the acting loop.

def run_dqn_episode(
    dqn_agent: DQNAgent,
    env: dm_catch.Catch,
    eval: bool,
) -> float:
  """Runs a single episode of catch.

  Args:
    dqn_agent: agent to train or evaluate
    env: the Catch environment the agent should interact with.
    eval: evaluation mode.
  Returns:
    The total reward accumulated over the episode.
  """
  # Reset any counts and start the environment.
  timestep = env.reset()
  dqn_agent.observe_first(timestep.observation)
  total_reward = 0

  # Run an episode.
  while not timestep.last():

    # Generate an action from the agent's policy and step the environment.
    action = dqn_agent.select_action(timestep.observation[None, ...], eval)
    timestep = env.step(action)

    # If the agent is training (not eval), add the transition to the replay
    # buffer and do an update step.
    if not eval:
      dqn_agent.observe(action, timestep)

    total_reward += timestep.reward

  return total_reward



# -------------------------------------------------------------------------------------------------   








# ------------------------------------------------------------------------------------------------

#@title Train the DQN agent. { form-width: "30%" }

num_episodes = 2_000
batch_size = 32

# Evaluation hyperparameters.
num_eval_episodes = 10
eval_every_period = 200

# Bind a dummy observation to the init function so the agent doesn't have to.
observation_spec = environment.observation_spec()
dummy_observation = np.zeros(observation_spec.shape, observation_spec.dtype)
init_params_fn = lambda rng: dqn_network.init(rng, dummy_observation[None, ...])

print(observation_spec)

# Create the agent.
dqn_agent = DQNAgent(
    init_params_fn=init_params_fn,
    network_apply_fn=dqn_network.apply,
    optimizer=optax.adam(learning_rate=3e-4),
    gamma=0.9,
    epsilon=0.3,
    num_actions=environment.action_spec().num_values,
    buffer_capacity=1_000,
    batch_size=batch_size,
    target_ema=0.99,
)

print(f"Episode number:\t| Average reward on {num_eval_episodes} eval episodes")
print("------------------------------------------------------")

# Initialize logged quantities.
episodes = []
all_rewards = []
all_losses = []
online_test_q_values = []
target_test_q_values = []


# TODO: I don't understand this.

# Create a test batch of all possible initial observations.
initial_glove_row = np.zeros((1, num_columns))
initial_glove_row[:, num_columns // 2] = 1
test_batch = np.concatenate(
    [
        np.zeros((num_columns, num_rows - 2, num_columns)),
        np.eye(num_columns)[:, None, :],
        num_columns * [initial_glove_row],
    ],
    axis=1,
)

for episode in range(num_episodes):
  # Run a training episode and then a training step.
  run_dqn_episode(
      dqn_agent,
      environment,
      eval=False
  )
  loss = dqn_agent.update()
  

  # Store some important diagnostic metrics to plot later.
  all_losses.append(loss)
  online_test_q_values.append(
      dqn_network.apply(dqn_agent._learner_state.online_params, test_batch)
  )
  target_test_q_values.append(
      dqn_network.apply(dqn_agent._learner_state.target_params, test_batch)
  )

  # Every once in a while, evaluate the greedy policy on a few episodes.
  if episode % eval_every_period == 0:
    reward = np.mean([
        run_dqn_episode(dqn_agent, environment, eval=True)
        for _ in range(num_eval_episodes)
    ])
    # Print how much reward the agent accumulated on average.
    print(f"\t{episode}\t|\t{reward}")
    all_rewards.append(reward)
    episodes.append(episode)
    















# ------------------------------------------------------------------------------   




















# ------------------------------------------------------------------------------   
# @title Visualize the evaluation return and the training loss.

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].set_xlabel('Number of training episodes')
axs[0].set_ylabel('Average return')
axs[0].plot(episodes, all_rewards)
axs[0].set_ylim([-1.1, 1.1])
axs[1].set_xlabel('Number of updates')
axs[1].set_ylabel('Average loss')
axs[1].plot(np.asarray(all_losses).T)
axs[1].set_ylim([0, 0.25]);

# ------------------------------------------------------------------------------

# @title Visualize the Q-values at all possible last steps.

online_test_q_values = np.asarray(online_test_q_values)
target_test_q_values = np.asarray(target_test_q_values)

fig, axs = plt.subplots(2, num_columns, figsize=(num_columns * 5, 10), sharey=True, sharex=True)
plt.ylim([-1.1, 1.1])
axs[0, 0].set_ylabel('Online Q-values')
axs[1, 0].set_ylabel('Target Q-values')
axs[1, 2].set_xlabel('Number of online network updates')
for i in range(num_columns):
  if i == num_columns // 2 - 1:
    should_play = 'left'
  elif i == num_columns // 2:
    should_play = 'no-op'
  elif i == num_columns // 2 + 1:
    should_play = 'right'
  else:
    should_play = None

  axs[0, i].set_title(f'Should play {should_play}' if should_play else None)
  axs[0, i].plot(online_test_q_values[:, i], label=['left', 'no-op', 'right'])
  axs[1, i].plot(target_test_q_values[:, i])

  if i == 0:
    axs[0, i].legend() 