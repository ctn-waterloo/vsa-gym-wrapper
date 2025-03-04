import gymnasium as gym

class FlatWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env: gym.Env,
            **kwargs
    ):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        return obs['image']
