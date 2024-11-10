import os
from typing import Any

import torch
import imageio
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from safety_robot_gym.envs.fetch import MujocoBlockedFetchPushEnv

gpu_id: int = 1
total_timesteps: int = 1_000_000

policy_size: str = "large"

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env_config: dict[str, Any] = {
    "render_mode": "rgb_array",
    "reward_type": "dense",
    "penalty_type": "dense",
    "dense_penalty_coef": 0.01,
    "sparse_penalty_value": -10,
    "max_episode_steps": 100,
}

policy_network_size_options: dict[str, dict[str, Any]] = {
    "default": {
        "net_arch": [256, 256],
    },
    "large": {
        "net_arch": [512, 512, 512],
        "n_critics": 2,
    },
}
file_title: str = (
    f"fetch_push_sac_{env_config['reward_type']}_reward_{env_config['penalty_type']}_penalty_{policy_size}"
)
model_save_path: str = f"out/models/{file_title}.zip"
animation_save_path: str = f"out/plots/{file_title}.gif"
tb_log_path: str = "out/logs/fetch_push"

from_check_point: bool = False

if from_check_point:
    ckpt_step: int = 1000_000
    model_save_path: str = f"out/models/ckpts/{file_title}_{ckpt_step}_steps.zip"
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_save_path}")
    else:
        print(f"Loading model from checkpoint: {model_save_path}")

sac_config: dict[str, Any] = {
    "policy": "MultiInputPolicy",
    "buffer_size": int(1e6),
    "batch_size": 1024,
    "gamma": 0.95,
    "learning_rate": 0.001,
    "learning_starts": 1000,
    "tau": 0.05,
    "tensorboard_log": tb_log_path,
    "policy_kwargs": policy_network_size_options[policy_size],
}
her_config: dict[str, Any] = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "copy_info_dict": True,
}

env = MujocoBlockedFetchPushEnv(**env_config)

if os.path.exists(model_save_path):
    model = SAC.load(model_save_path, env=env, device=device)
else:
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
        tb_log_name=file_title.removeprefix("fetch_push_"),
    )

demo_env = MujocoBlockedFetchPushEnv(
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

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=10)
