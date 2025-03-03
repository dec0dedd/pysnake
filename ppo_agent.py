from typing import List

from stable_baselines3 import PPO


class PPOAgent:
    def __init__(
            self,
            block_size: List[int],
            model_path: str
            ):
        self.block_size = block_size
        self.model_path = model_path
        self.model = PPO.load(self.model_path)

    def act(self, obs):
        return self.model.predict(obs)[0]
