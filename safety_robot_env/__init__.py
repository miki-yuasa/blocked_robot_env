import os
from gymnasium.envs.registration import register

register(
    id="safety_robot_env/MujocoBlockedFetchPickAndPlace-v0",
    entry_point="safety_robot_env.envs.fetch.pick_and_place:MujocoBlockedFetchPickAndPlaceEnv",
)
register(
    id="safety_robot_env/MujocoBlockedFetchPush-v0",
    entry_point="safety_robot_env.envs.fetch.push:MujocoBlockedFetchPushEnv",
    max_episode_steps=500,
    kwargs={
        "reward_type": "dense",
        "obstacle_penalty": True,
        "goal_reward": 10,
        "model_path": os.path.join("fetch", "blocked_push.xml"),
        "n_substeps": 20,
        "target_offset": 0.0,
        "obj_range": 0.15,
        "target_range": 0.15,
        "distance_threshold": 0.05,
        "render_mode": "rgb_array",
    },
)
