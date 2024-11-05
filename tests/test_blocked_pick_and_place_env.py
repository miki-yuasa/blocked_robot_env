import os

import imageio

from blocked_robot_env.envs.fetch.pick_and_place import (
    MujocoBlockedFetchPickAndPlaceEnv,
)


def test_env_init() -> None:
    animation_save_path: str = "tests/out/plots/test_blocked_pick_and_place_env.gif"
    os.environ["MUJOCO_GL"] = "osmesa"
    env = MujocoBlockedFetchPickAndPlaceEnv(
        render_mode="rgb_array",
    )

    obs, _ = env.reset()
    frames = [env.render()]

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())

        if terminated or truncated:
            break
        else:
            pass

    imageio.mimsave(animation_save_path, frames, fps=30)

    assert os.path.exists(animation_save_path)
