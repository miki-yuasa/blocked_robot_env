import os
from typing import Any, Literal

import torch
import imageio
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TQC

from safety_robot_gym.envs.fetch import (
    MujocoBlockedFetchPushEnv,
    MujocoBlockedFetchPickAndPlaceEnv,
)

gpu_id: int = 1
total_timesteps: int = 3_000_000

env_name: Literal["blocked_fetch_push", "blocked_fetch_pick_and_place"] = (
    "blocked_fetch_pick_and_place"
)
policy_size: str = "large"
algo: Literal["sac", "tqc"] = "sac"

restart_from_the_last_checkpoint: bool = False

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env_config: dict[str, Any] = {
    "render_mode": "rgb_array",
    "reward_type": "dense",
    "penalty_type": "dense",
    "dense_penalty_coef": 0.01,
    "sparse_penalty_value": 10,
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
    f"{env_name}_{algo}_{env_config['reward_type']}_reward_{env_config['penalty_type']}_penalty_{policy_size}"
)
model_save_path: str = f"out/models/{file_title}.zip"
animation_save_path: str = f"out/plots/{file_title}.gif"
tb_log_path: str = f"out/logs/blocked_fetch"


if restart_from_the_last_checkpoint:
    ckpt_dir: str = "out/models/ckpts"
    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = [
        f for f in ckpt_files if f.startswith(file_title) and f.endswith(".zip")
    ]
    timesteps: list[float] = [int(f.split("_")[-2]) for f in ckpt_files]
    ckpt_files = [f for _, f in sorted(zip(timesteps, ckpt_files))]
    if len(ckpt_files) > 0:
        last_ckpt = ckpt_files[-1]
        ckpt_step = int(last_ckpt.split("_")[-2])
        model_save_path = os.path.join(ckpt_dir, last_ckpt)
        print(f"Loading model from the last checkpoint: {model_save_path}")
    else:
        print("No checkpoint file found, starting from scratch")

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
tqc_config: dict[str, Any] = {
    "policy": "MultiInputPolicy",
    "buffer_size": int(1e6),
    "batch_size": 2048,
    "gamma": 0.95,
    "learning_rate": 0.001,
    "tau": 0.05,
    "tensorboard_log": tb_log_path,
    "policy_kwargs": policy_network_size_options[policy_size],
}
her_config: dict[str, Any] = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "copy_info_dict": True,
}

match env_name:
    case "blocked_fetch_push":
        env = MujocoBlockedFetchPushEnv(**env_config)
    case "blocked_fetch_pick_and_place":
        env = MujocoBlockedFetchPickAndPlaceEnv(**env_config)
    case _:
        raise ValueError(f"Unknown environment: {env_name}")

if os.path.exists(model_save_path):  # and not restart_from_the_last_checkpoint:
    model = SAC.load(model_save_path, env=env, device=device)
else:
    if restart_from_the_last_checkpoint:
        total_timesteps = total_timesteps - ckpt_step
        model = SAC.load(model_save_path, env=env, device=device)
    else:
        match algo:
            case "sac":
                model = SAC(
                    env=env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_config,
                    verbose=1,
                    device=device,
                    **sac_config,
                )
            case "tqc":
                model = TQC(
                    env=env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_config,
                    verbose=1,
                    device=device,
                    **tqc_config,
                )
            case _:
                raise ValueError(f"Unknown algorithm: {algo}")

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="out/models/ckpts",
        name_prefix=file_title,
        save_replay_buffer=True,
    )

    model.learn(
        total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=file_title,
        reset_num_timesteps=not restart_from_the_last_checkpoint,
    )

    model.save(model_save_path)

demo_env = MujocoBlockedFetchPushEnv(**env_config)
obs, _ = demo_env.reset()
frames = [demo_env.render()]

ep_reward: float = 0
rewards: list[float] = []
while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print("Observation:")
    print(obs)
    print(f"Reward:{reward}")
    ep_reward += reward
    rewards.append(reward)
    frames.append(demo_env.render())
    if terminated or truncated:
        break

print(f"Episode reward: {ep_reward}")
print(f"Rewards: {rewards}")

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=10)
