import os
from typing import Any

import torch
import imageio
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback

from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

gpu_id: int = 0
total_timesteps: int = 1_000_000

policy_size: str = "large"

restart_from_the_last_checkpoint: bool = False

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env_config: dict[str, Any] = {
    "render_mode": "rgb_array",
    "reward_type": "dense",
}
max_episode_steps: int = 100

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
    f"normal_fetch_pick_and_place_sac_{env_config['reward_type']}_reward_{policy_size}"
)
model_save_path: str = f"out/models/{file_title}.zip"
animation_save_path: str = f"out/plots/{file_title}.gif"
tb_log_path: str = "out/logs/fetch_pick_and_place"

from_check_point: bool = False

if from_check_point:
    ckpt_step: int = 1000_000
    model_save_path: str = f"out/models/ckpts/{file_title}_{ckpt_step}_steps.zip"
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_save_path}")
    else:
        print(f"Loading model from checkpoint: {model_save_path}")
elif restart_from_the_last_checkpoint:
    ckpt_dir: str = "out/models/ckpts"
    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = [
        f for f in ckpt_files if f.startswith(file_title) and f.endswith(".zip")
    ]
    ckpt_files = sorted(ckpt_files)
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
her_config: dict[str, Any] = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "copy_info_dict": True,
}

env = MujocoFetchPickAndPlaceEnv(**env_config)
env = TimeLimit(env, max_episode_steps=max_episode_steps)

if os.path.exists(model_save_path):  # and not restart_from_the_last_checkpoint:
    model = SAC.load(model_save_path, env=env, device=device)
else:
    if restart_from_the_last_checkpoint:
        total_timesteps = total_timesteps - ckpt_step
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
        tb_log_name=file_title.removeprefix("fetch_pick_and_place_"),
        reset_num_timesteps=not restart_from_the_last_checkpoint,
    )

    model.save(model_save_path)

demo_env = MujocoFetchPickAndPlaceEnv(**env_config)
demo_env = TimeLimit(demo_env, max_episode_steps=max_episode_steps)
obs, _ = demo_env.reset()
frames = [demo_env.render()]

ep_reward: float = 0
while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print("Observation:")
    print(obs)
    print(f"Reward:{reward}")
    ep_reward += reward
    frames.append(demo_env.render())
    if terminated or truncated:
        break

print(f"Episode reward: {ep_reward}")

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=10)
