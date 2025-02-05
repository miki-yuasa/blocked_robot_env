import os

import numpy as np
import gymnasium as gym
import imageio

import pytest

import safety_robot_env


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_blocked_env_init(num_obs) -> None:
    animation_save_path: str = f"tests/out/plots/test_blocked_push_env_{num_obs}.gif"
    # os.environ["MUJOCO_GL"] = "osmesa"
    env = gym.make("BlockedFetchPush-v0", max_episode_steps=100, num_obs=num_obs)

    obs, info = env.reset()
    frames = [env.render()]

    init_obj_pos = info["init_object_pos"]
    init_obs_pos = info["init_obstacle_pos"]

    # assert np.linalg.norm(init_obj_pos[:2] - init_obs_pos[:2]) >= 0.1

    step_count: int = 0
    for _ in range(2 * env._max_episode_steps):
        step_count += 1
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())

        if terminated or truncated:
            break
        else:
            pass

    assert step_count <= env._max_episode_steps

    os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
    imageio.mimsave(animation_save_path, frames, fps=30)

    assert os.path.exists(animation_save_path)


def test_default_env_init() -> None:
    animation_save_path: str = "tests/out/plots/test_default_push_env.gif"
    # os.environ["MUJOCO_GL"] = "osmesa"
    env = gym.make("FetchPush-v0")

    obs, _ = env.reset()
    frames = [env.render()]

    step_count: int = 0
    for _ in range(2 * env._max_episode_steps):
        step_count += 1
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())

        if terminated or truncated:
            break
        else:
            pass

    assert step_count <= env._max_episode_steps

    os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
    imageio.mimsave(animation_save_path, frames, fps=30)

    assert os.path.exists(animation_save_path)
