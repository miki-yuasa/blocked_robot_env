import os

import imageio

from safety_robot_env.envs.fetch.push import (
    MujocoBlockedFetchPushEnv,
)


def test_env_init() -> None:
    animation_save_path: str = "tests/out/plots/test_blocked_push_env.gif"
    os.environ["MUJOCO_GL"] = "osmesa"
    env = MujocoBlockedFetchPushEnv(
        render_mode="rgb_array",
        penalty_type="zero",
    )

    obs, _ = env.reset()
    frames = [env.render()]

    step_count: int = 0
    for _ in range(2 * env.max_episode_steps):
        step_count += 1
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())

        if terminated or truncated:
            break
        else:
            pass

    assert step_count <= env.max_episode_steps

    os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
    imageio.mimsave(animation_save_path, frames, fps=30)

    assert os.path.exists(animation_save_path)
