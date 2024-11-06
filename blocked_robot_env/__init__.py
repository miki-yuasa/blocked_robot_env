from gymnasium.envs.registration import register

register(
    id="blocked_robot_env/MujocoBlockedFetchPickAndPlace-v0",
    entry_point="blocked_robot_env.envs.fetch.pick_and_place:MujocoBlockedFetchPickAndPlaceEnv",
)
