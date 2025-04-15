
import gymnasium as gym

class gym_env_wrapper():
    def __init__(self, env_name, *kwargs):
        self.env = gym.make(env_name, *kwargs)
    
    def reset(self, seed=None):
        if seed is not None:
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()
        return state
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return state, reward, done, info
    
    def close(self):
        self.env.close()

    def set_seed(self, seed):
        self.env.seed = seed

    def get_seed(self):
        return self.env.seed

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]
    
    @property
    def action_dim(self):
        return self.env.action_space.n