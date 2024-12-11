import os
from typing import Any

import imageio
from gymnasium.wrappers import TimeLimit

from safety_robot_env.envs.fetch.push import MujocoFetchPushEnv


animation_save_path: str = "out/plots/test_default_push_env.gif"
os.environ["MUJOCO_GL"] = "osmesa"

env = MujocoFetchPushEnv(reward_type="dense", render_mode="rgb_array")

obs, _ = env.reset()
frames = [env.render()]

step_count: int = 0
for _ in range(50):
    step_count += 1
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frames.append(env.render())
    print(f"Step: {step_count}, Reward: {reward}")

    if terminated or truncated:
        break
    else:
        pass


os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30)

assert os.path.exists(animation_save_path)
