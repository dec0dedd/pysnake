from game_env import SnakeEnv
from ppo_callbacks import LogSnakeLength

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


env = make_vec_env(
    SnakeEnv,
    n_envs=4,
)

env = VecNormalize(
    env,
    norm_obs=False,
    norm_reward=True
)

model = PPO(
    "MultiInputPolicy",
    env=env,
    n_steps=256,
    batch_size=128,
    verbose=1,
    tensorboard_log='logs/ppo_logs',
    learning_rate=linear_schedule(1e-4),
    ent_coef=0.01,
)

TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 200_000

checkpoint_callback = CheckpointCallback(
  save_freq=SAVE_FREQ,
  save_path="models/ppo",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback, LogSnakeLength()])
