from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from datetime import datetime
import pickle
from os.path import exists
from environment.maze import Maze,  Render, Status

# maze = jnp.array( [
#     [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
# ])

maze = np.array([
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
# maze = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0]
# ])

game = Maze(maze)

class SARSAModel():
    
    def __init__(self,
                 game, 
                 model_name= "data.pkl",
                 size: int = 20):
        # super().__init__(game, name="SARSAModel", **kwargs)
        # creating the Q dictionary of key: (state, action) and value: [0,1]
        self.Q= dict()
        self.size = size
        if exists(f"{self.size}/{model_name}"):
            self.Q = self.load_model(f"{self.size}/{model_name}")
        self.environment = game
        self.name = "SARSA"
        self.key = jax.random.PRNGKey(0)
        
        
        
    def train(self, **kwargs):
        # params passed throug the **kwargs are
        # 1. discount: float64
        # 2. exploration_rate: float64
        # 3. learning_rate: float64
        # 4. episodes: int
        # 5. datetime: int
        
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.997)
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        
        # check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)
        
        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []
        
        start_list = list()
        start_time = datetime.now()
        timelapse = None
        for episode in range(1, episodes+1):
            if not start_list:
                start_list = self.environment.empty.copy()
            # setting a random starting cell and removing it from the start_list
            randint = jax.random.randint(jax.random.PRNGKey(0), shape=(1,), minval=0, maxval=len(start_list)).astype(jnp.uint8)[0]
            start_cell = start_list[randint]
            start_list.remove(start_cell)
            
            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())
            
            if jax.random.uniform(key=self.key, shape=(1,))[0] < exploration_rate:
                action = jax.random.choice(key=self.key, a=self.environment.actions)
            else:
                action = self.predict(self.Q, state)
            while True:
                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                next_action = self.predict(self.Q, next_state)
                
                cumulative_reward +=reward
                action = float(action)
                if (self.makeHashable(state), action) not in self.Q:
                    self.Q[(self.makeHashable(state), action)]=0.0
                    
                next_q= self.Q.get((self.makeHashable(next_state), float(next_action)), 0.0)
                
                # Updating the Q value of 
                self.Q[(self.makeHashable(state), float(action))] += learning_rate * (reward + discount * next_q -self.Q[(self.makeHashable(state), float(action))])
                
                if status in (Status.WIN, Status.LOSE):
                    break 
                # Resetting the next state values to current.
                state = next_state
                action = next_action
                self.environment.render_q(self)
                
            # checks for convergence every 5 episode.
            if episode % 5 == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                
                # This will stop at convergence.
                if w_all:
                    print("won from all start cells, stop learning\n")
                    break
                exploration_rate *= exploration_decay
            print(f"Episod: {episode}/{episodes} | status: {status.name} |e-val: {exploration_rate}\n")
        self.dump_model(self.Q)
        timelapse = (datetime.now() - start_time).total_seconds()
        print("Time taken to finish: ", timelapse, " seconds")
        return cumulative_reward_history, win_history, episode, datetime.now()-start_time
    
    def makeHashable(self, state):
        if isinstance(state, (jnp.ndarray, np.ndarray)):
            return tuple(state.flatten().tolist())
        elif isinstance(state, list):
            return tuple(state)
        return tuple((int(state[0]), int(state[1])))
        
    def dump_model(self, model: dict, name="data.pkl"):
        a_file = open(f"{self.environment.maze.shape[0]}/{name}", "wb")
        pickle.dump(model, a_file)
        a_file.close()
    
    def load_model(self, name="data.pkl") -> dict:
        d_file = open(name, "rb")
        output = pickle.load(d_file)
        d_file.close()
        return output
    
    def q(self, Q_dict, state):
        # Rrtuens a list of actions values for the provided state.
        state = self.makeHashable(state)
        q_values = [
            Q_dict.get((state, action), 0.0) 
            for action in self.environment.actions
            ]
        return jnp.array(q_values)
        

    def predict(self, Q_dict, state):
        # Chosing which action we are going to take.
        qvals = self.q(Q_dict, state)
        actions = jnp.where(qvals == jnp.max(qvals))[0]
        return jax.random.choice(key=self.key, a=actions)


model = SARSAModel(game)
game.play(model)

# Train the model
cumulative_reward_history, win_history, episode, training_duration=model.train(discount=0.9, exploration_rate=0.1, exploration_decay=0.999, learning_rate=0.1, episodes=30000000)
try:
    cumulative_reward_history  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.manager.set_window_title(model.name)
    ax1.plot(*zip(*win_history))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    # ax2.plot(h)
    ax2.plot(cumulative_reward_history)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

game.render(Render.MOVES)
game.play(model, start_cell=(4, 1))
plt.show()
