import os
from gymnasium.envs.registration import register

register(
    id="BlockedFetchPickAndPlace-v0",
    entry_point="safety_robot_env.envs.fetch.pick_and_place:MujocoBlockedFetchPickAndPlaceEnv",
)
register(
    id="BlockedFetchPush-v0",
    entry_point="safety_robot_env.envs.fetch.push:MujocoBlockedFetchPushEnv",
    max_episode_steps=50,
    kwargs={
        "reward_type": "dense",
        "obstacle_penalty": "step",
        "num_obs": 1,
        "penalty_scale": 1.0,
        "n_substeps": 20,
        "obj_range": 0.15,
        "target_range": 0.15,
        "obj_clearance": 0.1,
        "obs_clearance": 0.1,
        "distance_threshold": 0.05,
        "terminate_upon_success": False,
        "terminate_upon_collision": False,
        "render_mode": "rgb_array",
    },
)

register(
    id="FetchPush-v0",
    entry_point="safety_robot_env.envs.fetch.push:MujocoFetchPushEnv",
    max_episode_steps=50,
    kwargs={"reward_type": "dense", "render_mode": "rgb_array"},
)
