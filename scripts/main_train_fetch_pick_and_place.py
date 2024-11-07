import os
from typing import Any

import torch
import imageio
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from blocked_robot_env.envs.fetch import MujocoBlockedFetchPickAndPlaceEnv

gpu_id: int = 1
total_timesteps: int = 1_000_000

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env_config: dict[str, Any] = {
    "render_mode": "rgb_array",
    "reward_type": "dense",
    "penalty_type": "dense",
    "dense_penalty_coef": 0.1,
    "sparse_penalty_value": -100,
    "max_episode_steps": 100,
}

file_title: str = (
    f"fetch_pick_and_place_sac_reward_{env_config['reward_type']}_penalty_{env_config['penalty_type']}"
)
model_save_path: str = f"out/models/{file_title}.zip"
animation_save_path: str = f"out/plots/{file_title}.gif"
tb_log_path: str = "out/logs/fetch_pick_and_place"

sac_config: dict[str, Any] = {
    "policy": "MultiInputPolicy",
    "buffer_size": int(1e6),
    "batch_size": 1024,
    "gamma": 0.95,
    "learning_rate": 0.001,
    "learning_starts": 1000,
    "tau": 0.05,
    "tensorboard_log": tb_log_path,
}
her_config: dict[str, Any] = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "copy_info_dict": True,
}

env = MujocoBlockedFetchPickAndPlaceEnv(**env_config)

model = SAC(
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=her_config,
    verbose=1,
    device=device,
    **sac_config,
)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="out/models/ckpts",
    name_prefix=file_title,
    save_replay_buffer=True,
)

model.learn(
    total_timesteps,
    callback=checkpoint_callback,
    tb_log_name=file_title.removeprefix("fetch_pick_and_place_"),
)

demo_env = MujocoBlockedFetchPickAndPlaceEnv(
    render_mode="rgb_array",
)
obs, _ = demo_env.reset()
frames = [demo_env.render()]

while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print("Observation:")
    print(obs)
    print(f"Reward:{reward}")
    frames.append(demo_env.render())
    if terminated or truncated:
        break

demo_env.close()

imageio.mimsave(animation_save_path, frames, fps=30, loop=10)
