import chex
import tree
import haiku as hk
import random
import optax
import numpy as np
import environment
from environment.maze import Maze, Status
import jax.numpy as jnp


@chex.dataclass
class Transition:
    state: chex.Array # the state of the cell
    action: chex.Array # What action was taken when in the above observation
    reward: chex.Array # reward we get after taking this action above
    done: chex.Array # Is it the last state
    
    next_state: chex.Array # the next state



class ReplayByuffer:
    def __init__(self, max_mem):
        self._max_mem = max_mem
        self.memory = []
    
    @property
    def size(self) -> int: # returns the length when accessed as a method or as a field
        return len(self.memory)
    
    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self._max_mem:
            self.memory.pop(0)
    
    def sample_batch(self, batch_size):
        assert len(self.memory) >= batch_size, "Not enough samples in the memory"
        transitions: list[Transition] = random.choices(self.memory, k=batch_size)
        return tree.map_structure(lambda *x: np.stack(x), *transitions)
    

# ----------------------------------------------------------------

# Where to store our online and target models?

@chex.dataclass  # specifying the class is for storing the data
class LearnerState:
    online_params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState # The optimizer
    
# ----------------------------------------------------------------




# An abstract class to represent DDQN and Dueling DQN Agents.
class AbstractAgent:
    
    def predict(self, state) -> int:
        pass
    
    def dump_model(self, params: hk.Params,model_name="directory/filename.pkl"):
        pass
    def _update_function(
                self,
                state: LearnerState,
                batch: Transition,
                ) -> tuple[LearnerState, chex.Array]:
        pass
    def select_action(self, 
                      state: chex.Array, 
                      eval: bool) -> chex.Array:
         pass
    
    def first_state(self, state: chex.Array):
        pass 
    
    def _loss_function(self, 
                       online_params: hk.Params,
                       target_params: hk.Params,
                       transitions: Transition
                       ) -> chex.Array: # returns a loss
        pass
    
    def observe(self, state, action, next_state, reward, status):
        pass
    
    def update(self) -> chex.Array | None:
        pass
    def buffer_size(self)->int:
        return 0, 0
    

# runs one episode and collects the total reward it got.
# If Eval it is to see the total cumulative reward
# else if Not Eval it is to collect the moves and use that for training.
def run_dqn_episode(
    dqn_agent: AbstractAgent,
    env: Maze,
    eval: bool,
) -> float:
    """
    Runs a single episode of catch.

    Args:
        dqn_agent: agent to train or evaluate
        env: the Catch environment the agent should interact with.
        eval: evaluation mode.
        Returns:
        The total reward accumulated over the episode.
    """
    # variables for reporting purposes
    cumulative_reward = 0

    start_list = list()  # starting cells not yet used for training
    
    if not start_list:
        start_list = env.empty.copy()
    start_cell = random.choice(start_list)
    start_list.remove(start_cell)
    
    # reseting the environment by giving a starting cell (state)
    env.reset(start_cell)
    
    start_cell = jnp.asarray(start_cell)
    
    dqn_agent.first_state(start_cell)
    state = start_cell
    maxsteps = env.maze.shape[0]**2
    step =0
    while True:
        
        # Generate an action from the agent's policy and step the environment.
        action = dqn_agent.select_action(state[None, ...], eval)
        next_state, reward, status = env.step(action)
        next_state = jnp.squeeze(next_state, axis=0)
        
        next_state = jnp.asarray(next_state)
        
        # If the agent is training (not eval), add the transition to the replay
        # buffer and do an update step.
        if not eval:
            dqn_agent.observe(state, action,next_state, reward, status)
            
        cumulative_reward += reward
        step +=1
        if (status in (Status.LOSE, Status.WIN)) or (step >= maxsteps):
            break
        state = next_state
    buffer_size, batch_size = dqn_agent.buffer_size()
    if buffer_size < batch_size:
        cumulative_reward += run_dqn_episode(dqn_agent, env,eval)
    return cumulative_reward