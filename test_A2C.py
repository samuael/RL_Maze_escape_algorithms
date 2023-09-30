from environment import Maze, Status
import jax
import jax.numpy as jnp
import haiku as hk
import chex
from typing import Callable
import optax
import jaxlib
import numpy as np
    

@chex.dataclass
class Transition:
    state: chex.Array
    log_prob: chex.Array  # The log probability of taking action.
    next_state: chex.Array
    reward: chex.Array
    done: chex.Array

def actorNetwork(x: chex.Array):
    out = hk.Linear(128)(x)
    out = jax.nn.relu(out)
    out = hk.Linear(64)(out)
    out = jax.nn.relu(out)
    return hk.Linear(4)(out)


def criticNetwork(x: chex.Array):
    out = hk.Linear(648)(x)
    out = jax.nn.relu(out)
    return hk.Linear(1)(out)
    

 
@chex.dataclass  # specifying the class is for storing the data
class A2CLearnerState:
    actor_param: hk.Params
    critic_param: hk.Params
    a_opt_state: optax.OptState # The optimizer state
    c_opt_state: optax.OptState # The optimizer state

class A2CAgent:

    def __init__(self, env: Maze, 
                 actor_init_func: Callable[[jax.random.KeyArray], hk.Params],
                 actor_apply_func: Callable[[hk.Params, chex.Array], chex.Array],
                 critic_init_func: Callable[[jax.random.KeyArray], hk.Params],
                 critic_apply_func: Callable[[hk.Params, chex.Array], chex.Array],
                 actor_optimizer: optax.GradientTransformation,
                 critic_optimizer: optax.GradientTransformation,
                 gamma: float,
                 entropy_weight: float, 
                 seed: int=0,):
        """Initialize."""
        self.env = env
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        
        
        self.actor_apply = actor_apply_func
        self.critic_apply = critic_apply_func
        
        actor_param = actor_init_func(jax.random.PRNGKey(seed))
        print(actor_param)
        critic_param = critic_init_func(jax.random.PRNGKey(seed))
        print(critic_param)
        
        self.state = A2CLearnerState(
            actor_param = actor_param,
            critic_param = critic_param,
            a_opt_state = self._actor_optimizer.init(actor_param),
            c_opt_state = self._critic_optimizer.init(critic_param),
        )
        
        # transition (state, log_prob, next_state, reward, done)
        self.transition: Transition = None
        self.total_step = 0
        self.eval = False
        
        self._update = jax.jit(self._update_func)
        
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
        if eval or np.random.uniform() > 0.11:
            state_expand = jnp.expand_dims(state, axis=0)
            # Greedy action selection
            q_values = self.actor_apply(
                self.state.actor_param,
                state_expand,
            )
            q_values = jnp.squeeze(q_values, axis=0)
            val = jnp.argmax(q_values, axis=-1)
            if isinstance(val, list) or isinstance(val, jaxlib.xla_extension.ArrayImpl):
                return int(val[0]), jnp.squeeze(q_values)
            return int(val), q_values
        val = np.random.randint(4)
        return val, jnp.squeeze(jnp.asarray(self.int_to_one_hot(val)))
    
    def step(self, action: chex.Array, eval:bool = False): # -> tuple((chex.Array, chex.Array, bool)):
        """Take an action and return the response of the env."""
        next_state, reward, status = self.env.step(action)
        return next_state, reward, (status in [Status.WIN, Status.LOSE])
    
    def _critic_loss_function(self, 
                       critic_params: hk.Params, 
                       transition: Transition,
                       )-> chex.Array:

        mask = 1 - self.transition.done
        if isinstance(self.transition.state, tuple):
            self.transition.state= jnp.asarray(self.transition.state)
        
        next_qs = jnp.argmax(self.critic_apply(critic_params,transition.state), axis=-1, keepdims=True)    
        
        y = jnp.where(
            transition.done,
            transition.reward,
            transition.reward + self.gamma * (self.critic_apply(critic_params,transition.next_state)).squeeze(),
        )
        return 0.5 * jnp.mean(jnp.square(next_qs - y))
    
    def _critic_loss_advantage(self, 
                       critic_params: hk.Params,
                       transition: Transition,
                       ): #-> chex.Array:

        mask = 1 - self.transition.done
        if isinstance(self.transition.state, tuple):
            self.transition.state= jnp.asarray(self.transition.state)
        
        next_qs = jnp.argmax(self.critic_apply(critic_params,transition.state), axis=-1, keepdims=True)    
        
        target = jnp.where(
            transition.done,
            transition.reward,
            transition.reward + self.gamma * jnp.argmax(self.critic_apply(critic_params,transition.next_state), axis=-1, keepdims=True),
        )

        # jitted = jax.jit(self.subtract, static_argnums=(0,1))
        target = jnp.squeeze(target)
        next_qs= jnp.squeeze(next_qs)
        return jax.lax.stop_gradient(float(target- next_qs))
    
    def int_to_one_hot(self,integer):
        return jnp.array(integer == jnp.arange(4), dtype=jnp.float32)
    def _actor_loss_func(self, 
                         actor_params: hk.Params,
                         transition: Transition,
                         ) -> chex.Array:
        advantage = self._critic_loss_advantage(self.state.critic_param, transition=transition)
        log_probs = transition.log_prob
        action = jnp.argmax(log_probs, axis=-1)
        log_probs = jnp.log(log_probs[action]+0.00001)
        policy_loss = -advantage * log_probs
        return policy_loss

    def _update_func(self, transition: Transition,
                     ):# -> tuple(chex.Array, chex.Array):
        if isinstance(self.transition.state, tuple):
            self.transition.state= jnp.squeeze(jnp.asarray(self.transition.state))
        if isinstance(self.transition.next_state, tuple):
            self.transition.state= jnp.squeeze(jnp.asarray(self.transition.next_state))
        
        policy_loss, a_gradient = jax.value_and_grad(self._actor_loss_func)(
            self.state.actor_param,
            transition=transition,
        )
        a_gradient = jax.grad(lambda actor_params: policy_loss)(self.state.actor_param)
        aupdates, aopt_state = self ._actor_optimizer.update(a_gradient, self.state.a_opt_state)
        actor_params = optax.apply_updates(self.state.actor_param, aupdates)
        
        value_loss, gradient = jax.value_and_grad(self._critic_loss_function)(
            self.state.critic_param,
            transition=transition,
        )
        
        updates, copt_state = self._critic_optimizer.update(gradient, self.state.c_opt_state)
        critic_params = optax.apply_updates(self.state.critic_param, updates)
        
        self.state = A2CLearnerState(
            actor_param = actor_params,
            critic_param = critic_params,
            a_opt_state = aopt_state,
            c_opt_state = copt_state,
        )
        return policy_loss, value_loss
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        actor_losses, critic_losses, scores = [], [], []
        state = self.env.reset()
        score = 0
        
        for self.total_step in range(1, num_frames + 1):
            action, log_probs = self.select_action(state, eval=False)
            next_state, reward, done = self.step(action)
            
            self.transition = Transition(
                state=state,
                log_prob=log_probs,
                next_state=next_state,
                reward=reward,
                done= done,
            )
            
            actor_loss, critic_loss = self._update_func(self.transition)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            state = next_state
            score += reward
            
            # if episode ends
            if state in [Status.WIN, Status.LOSE]:         
                state = self.env.reset()
                scores.append(score)
                score = 0                
            
            # plot
            if self.total_step % plotting_interval == 0:
                pass
                # self._plot(self.total_step, scores, actor_losses, critic_losses)
        self.env.close()
    
    def test(self):
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        self.env.close()
        
        return frames
        

num_frames = 100000
gamma = 0.9
entropy_weight = 1e-2


default_env = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
])
env = Maze(default_env)

# removes the random generator parameter from the apply() function.
actor_network = hk.without_apply_rng(hk.transform(actorNetwork))

critic_network = hk.without_apply_rng(hk.transform(criticNetwork))

# Bind a dummy observation to the init function so the agent doesn't have to.
dummy_observation = jnp.zeros((2,), float)
ainit_params_fn = lambda rng: actor_network.init(rng, dummy_observation[None, ...])

dummy_observation = jnp.zeros((2,), float)
cinit_params_fn = lambda rng: critic_network.init(rng, dummy_observation[None, ...])


if __name__ == '__main__':
    agent = A2CAgent(  
        env=env,  
        actor_init_func=ainit_params_fn,
        actor_apply_func=actor_network.apply,
        critic_init_func=cinit_params_fn,
        critic_apply_func=critic_network.apply,
        actor_optimizer=optax.adam(learning_rate=1e-4),
        critic_optimizer=optax.adam(learning_rate=1e-3),
        gamma=gamma,
        entropy_weight=entropy_weight,)
    agent.train(num_frames)