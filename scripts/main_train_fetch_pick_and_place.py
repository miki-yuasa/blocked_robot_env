import os
from typing import Any

import torch
import imageio
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

from blocked_robot_env.envs.fetch import MujocoBlockedFetchPickAndPlaceEnv

gpu_id: int = 1
total_timesteps: int = int(1e6)
model_save_path: str = "out/models/fetch_pick_and_place_sac"
animation_save_path: str = "out/plots/fetch_pick_and_place_sac.gif"
tb_log_path: str = "out/logs/fetch_pick_and_place"

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

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

env = MujocoBlockedFetchPickAndPlaceEnv(
    render_mode="rgb_array",
)

model = SAC(
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=her_config,
    verbose=1,
    device=device,
    **sac_config,
)

model.learn(
    total_timesteps,
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
