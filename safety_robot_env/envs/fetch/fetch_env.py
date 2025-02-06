from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray

from gymnasium_robotics.envs.fetch.fetch_env import (
    goal_distance,
    DEFAULT_CAMERA_CONFIG,
)
from gymnasium_robotics.utils import rotations

from safety_robot_env.envs.robot_env import MujocoRobotEnv


class BaseFetchEnv(MujocoRobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        gripper_extra_height: float,
        block_gripper: bool,
        has_object: bool,
        target_in_the_air: bool,
        target_offset: float | NDArray[np.float64],
        obj_range: float,
        target_range: float,
        obj_clearance: float,
        obs_clearance: float,
        distance_threshold: float,
        reward_type: Literal["sparse", "dense"],
        **kwargs,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.gripper_extra_height: float = gripper_extra_height
        self.block_gripper: bool = block_gripper
        self.has_object: bool = has_object
        self.target_in_the_air: bool = target_in_the_air
        self.target_offset: float | NDArray[np.float64] = target_offset
        self.obj_range: float = obj_range
        self.target_range: float = target_range
        self.obj_clearance: float = obj_clearance
        self.obs_clearance: float = obs_clearance
        self.distance_threshold: float = distance_threshold
        self.reward_type: Literal["sparse", "dense"] = reward_type

        super().__init__(n_actions=4, **kwargs)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        return action

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def generate_mujoco_observations(self):

        raise NotImplementedError

    def _get_gripper_xpos(self):

        raise NotImplementedError

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


class MujocoFetchEnv(BaseFetchEnv):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = (
                np.zeros(0)
            )
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]


class MujocoBlockedFetchEnv(MujocoFetchEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        obstacle_penalty: Literal["step", "cumulative"] = "step",
        num_obs: int = 1,
        penalty_scale: float = 0.0,
        terminate_upon_success: bool = True,
        terminate_upon_collision: bool = False,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        """Initializes a new Fetch environment.

        Parameters
        ----------
        model_path : string
            Path to the environments XML file
        n_substeps : int
            Number of substeps the simulation runs on every call to step
        gripper_extra_height : float
            Additional height above the table when positioning the gripper
        block_gripper : bool
            Whether or not the gripper is blocked (i.e. not movable) or not
        has_object : bool
            Whether or not the environment has an object
        target_in_the_air : bool
            Whether or not the target should be in the air above the table or on the table surface
        target_offset : float or array with 3 elements
            Offset of the target
        obj_range : float
            Range of a uniform distribution for sampling initial object positions
        target_range : float
            Range of a uniform distribution for sampling a target
        obj_clearance : float = 0.1
            The minimum distance between the object and the gripper when resetting the environment.
        distance_threshold : float
            The threshold after which a goal is considered achieved
        initial_qpos : dict
            A dictionary of joint names and values that define the initial configuration
        reward_type : typing.Literal['sparse', 'dense'] = 'sparse'
            The reward type, i.e. sparse or dense
        penalty_scale : float = 0.0
            The scale factor to apply to the obstacle penalty.
            It is used as a multiplier to the obstacle displacement e.g., penalty = penalty_scale * displacement.
        terminate_upon_success : bool = True
            Whether the environment is terminated upon success.
        terminate_upon_collision : bool = False
            Whether the environment is terminated upon collision.
        """

        self.penalty_scale: float = penalty_scale
        self.obstacle_penalty: Literal["step", "cumulative"] = obstacle_penalty
        self.num_obs: int = num_obs
        self.terminate_upon_success: bool = terminate_upon_success
        self.terminate_upon_collision: bool = terminate_upon_collision

        self.init_obstacle_pos: list[NDArray[np.float64]] = [
            np.zeros(3) for _ in range(self.num_obs)
        ]
        self.cumulative_obstacle_displacement: NDArray[np.float64] = np.zeros(3)
        self.step_obstacle_displacement: NDArray[np.float64] = np.zeros(3)
        self.obs_names: list[str] = [f"object{i+1}" for i in range(self.num_obs)]

        super().__init__(default_camera_config=default_camera_config, **kwargs)

        if self.reward_type not in ["sparse", "dense"]:
            raise ValueError("Invalid reward type. Must be either 'sparse' or 'dense'.")
        else:
            pass

    # GoalEnv methods
    # ----------------------------

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float64],
        goal: NDArray[np.float64],
        info: dict[str, Any] | list[dict[str, Any]],
    ):
        # Compute distance between goal and the achieved goal.
        achieved_goal = achieved_goal.reshape(-1, 6)
        goal = goal.reshape(-1, 6)
        pos_achieved_goal: NDArray[np.float64] = achieved_goal[:, :3]
        pos_desired_goal: NDArray[np.float64] = goal[:, :3]
        disp_achieved_goal: NDArray[np.float64] = achieved_goal[:, 3:]
        disp_desired_goal: NDArray[np.float64] = goal[:, 3:]
        d: NDArray[np.float64] = goal_distance(pos_achieved_goal, pos_desired_goal)

        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            reward: NDArray[np.float64] = -d
            # reward += self.penalty_scale * (d < self.distance_threshold).astype(
            #     np.float64
            # )
            displacements: NDArray[np.float64] = goal_distance(
                disp_achieved_goal, disp_desired_goal
            )
            # reward -= self.penalty_scale * (
            #     displacements > self.distance_threshold
            # ).astype(np.float64)
            reward -= self.penalty_scale * displacements

            if reward.size == 1:
                return reward.item()
            else:
                return reward

    # RobotEnv methods
    # ----------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: int | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        obs, _ = super().reset(seed=seed, options=options)
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> dict[str, NDArray[np.float64]]:
        dict_obs: dict[str, NDArray[np.float64]] = self.generate_mujoco_observations()
        if self.num_obs != 1:
            obs = np.concatenate(
                [value.ravel() for value in dict_obs.values()]
                + [self.step_obstacle_displacement.ravel()]
            )
        else:
            obs_name: str = self.obs_names[0]
            obs = np.concatenate(
                [
                    dict_obs["grip_pos"].ravel(),
                    dict_obs["object_pos"].ravel(),
                    dict_obs[obs_name + "_pos"].ravel(),
                    dict_obs["object_rel_pos"].ravel(),
                    dict_obs[obs_name + "_rel_pos"].ravel(),
                    self.step_obstacle_displacement.ravel(),
                    dict_obs["gripper_state"],
                    dict_obs["object_rot"].ravel(),
                    dict_obs[obs_name + "_rot"].ravel(),
                    dict_obs["object_velp"].ravel(),
                    dict_obs["object_velr"].ravel(),
                    dict_obs[obs_name + "_velp"].ravel(),
                    dict_obs[obs_name + "_velr"].ravel(),
                    dict_obs["grip_velp"].ravel(),
                    dict_obs["gripper_vel"].ravel(),
                ]
            )

        displacement: NDArray[np.float64] = np.zeros(3)
        if self.obstacle_penalty == "cumulative":
            displacement = self.cumulative_obstacle_displacement
        elif self.obstacle_penalty == "step":
            displacement = self.step_obstacle_displacement
        else:
            raise ValueError(
                "Invalid obstacle penalty type. Must be either 'cumulative' or 'step'."
            )

        achieved_goal = np.concatenate(
            [dict_obs["object_pos"].copy(), displacement.copy()]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _get_dict_obs(self) -> dict[str, NDArray[np.float64]]:

        dict_obs: dict[str, NDArray[np.float64]] = self.generate_mujoco_observations()

        return dict_obs

    def _get_info(self) -> dict[str, Any]:
        info: dict[str, NDArray[np.float64]] = self._get_dict_obs()
        normal_obs: dict[str, NDArray[np.float64]] = self._get_obs()
        info["is_success"] = self._is_success(normal_obs["achieved_goal"], self.goal)
        info["init_object_pos"] = self.init_object_pos
        info["init_obstacle_pos"] = self.init_obstacle_pos
        info["goal"] = self.goal

        return info

    def compute_terminated(
        self,
        achieved_goal: NDArray[np.float64],
        desired_goal: NDArray[np.float64],
        info: dict[str, Any] = {},
    ) -> bool:
        """
        Termination criterion for the environment.
        If the distance between the achieved goal and the desired goal is less than the distance threshold, the environment is terminated.
        `self.terminate_upon_success` determines whether the environment is terminated upon success.
        If the obstacle has moved more than the distance threshold, the environment is terminated.
        `self.terminate_upon_collision` determines whether the environment is terminated upon collision.

        Parameters
        ----------
        achieved_goal: np.typing.NDArray[np.float64]
            The achieved goal position.
        desired_goal: np.typing.NDArray[np.float64]
            The desired goal position.
        info: dict[str, Any]
            Additional information about the environment.
            Not used in this method.

        Returns
        -------
        terminated: bool
            Whether the environment is terminated.
        """
        is_success: bool = (
            goal_distance(achieved_goal, desired_goal) < self.distance_threshold
        ) and self.terminate_upon_success
        obstacle_moved: bool = (
            np.linalg.norm(self.step_obstacle_displacement) > self.distance_threshold
        ) and self.terminate_upon_collision
        terminated: bool = is_success or obstacle_moved
        return terminated

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3]
            # Ensure goal is not too close to the obstacle
            obstacles_clear: bool = False
            while not obstacles_clear:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )

                for obs_pos in self.init_obstacle_pos:
                    if np.linalg.norm(goal[:2] - obs_pos[:2]) <= self.obs_clearance:
                        break
                    else:
                        obstacles_clear = True

            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        # Extend the goal to include cumulative displacement of the obstacle to the goal in the x, y, and z directions
        goal = np.concatenate([goal, np.zeros(3)])
        return goal.copy()

    def _step_callback(self):

        super()._step_callback()

        # Record the displacement of the obstacle
        step_obstacle_displacement: NDArray[np.float64] = np.zeros(3)
        curr_obs_pos_list: list[NDArray[np.float64]] = []
        for i, obs_name in enumerate(self.obs_names):
            curr_obstacle_pos: NDArray[np.float64] = self._utils.get_site_xpos(
                self.model, self.data, obs_name
            )
            step_obstacle_displacement += curr_obstacle_pos - self.prev_obstacle_pos[i]
            curr_obs_pos_list.append(curr_obstacle_pos)

        self.cumulative_obstacle_displacement += step_obstacle_displacement
        self.step_obstacle_displacement = step_obstacle_displacement
        self.prev_obstacle_pos = curr_obs_pos_list

    def generate_mujoco_observations(self) -> dict[str, NDArray[np.float64]]:
        # positions
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = super().generate_mujoco_observations()

        output_dict: dict[str, NDArray[np.float64]] = {
            "grip_pos": grip_pos,
            "object_pos": object_pos,
            "object_rel_pos": object_rel_pos,
            "gripper_state": gripper_state,
            "object_rot": object_rot,
            "object_velp": object_velp,
            "object_velr": object_velr,
            "grip_velp": grip_velp,
            "gripper_vel": gripper_vel,
        }

        if self.has_object:
            for obs_name in self.obs_names:
                # Add the obstacle position, rotation, velocity, and angular velocity to the observation (from "object1")
                dt = self.n_substeps * self.model.opt.timestep
                obstacle_pos: NDArray[np.float64] = self._utils.get_site_xpos(
                    self.model, self.data, obs_name
                )
                obstacle_rot: NDArray[np.float64] = rotations.mat2euler(
                    self._utils.get_site_xmat(self.model, self.data, obs_name)
                )
                obstacle_velp = (
                    self._utils.get_site_xvelp(self.model, self.data, obs_name) * dt
                )
                obstacle_velr = (
                    self._utils.get_site_xvelr(self.model, self.data, obs_name) * dt
                )
                # gripper state
                obstacle_rel_pos = obstacle_pos - grip_pos
                obstacle_velp -= grip_velp

                output_dict[obs_name + "_pos"] = obstacle_pos
                output_dict[obs_name + "_rot"] = obstacle_rot
                output_dict[obs_name + "_velp"] = obstacle_velp
                output_dict[obs_name + "_velr"] = obstacle_velr
                output_dict[obs_name + "_rel_pos"] = obstacle_rel_pos
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = (
                np.zeros(0)
            )

        return output_dict

    def _reset_sim(self) -> bool:
        super()._reset_sim()

        # Randomize start position of object.
        if self.has_object:

            # Randomize start position of obstacle.
            object_xpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )[:2]

            placed_obstacle_xpos_list: list[NDArray[np.float64]] = []

            init_obs_pos_list: list[NDArray[np.float64]] = []

            for obs_name in self.obs_names:
                obstacle_xpos = self.initial_gripper_xpos[:2]
                cleared_with_placed_obstacles: bool = False
                # Ensure obstacle is not placed on top of object and is not too close to the gripper
                while (
                    np.linalg.norm(obstacle_xpos - self.initial_gripper_xpos[:2])
                    < self.obs_clearance
                    or np.linalg.norm(obstacle_xpos - object_xpos) < self.obj_clearance
                    or not cleared_with_placed_obstacles
                ):
                    obstacle_xpos = self.initial_gripper_xpos[
                        :2
                    ] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

                    if len(placed_obstacle_xpos_list) > 0:
                        cleared_with_placed_obstacles = True
                        for placed_obstacle_xpos in placed_obstacle_xpos_list:
                            if (
                                np.linalg.norm(obstacle_xpos - placed_obstacle_xpos)
                                < self.obs_clearance
                            ):
                                cleared_with_placed_obstacles = False
                    else:
                        cleared_with_placed_obstacles = True
                obstacle_qpos = self._utils.get_joint_qpos(
                    self.model, self.data, f"{obs_name}:joint"
                )
                assert obstacle_qpos.shape == (7,)
                obstacle_qpos[:2] = obstacle_xpos
                self._utils.set_joint_qpos(
                    self.model, self.data, f"{obs_name}:joint", obstacle_qpos
                )
                placed_obstacle_xpos_list.append(obstacle_xpos)
                init_obs_pos_list.append(
                    self._utils.get_joint_qpos(
                        self.model, self.data, f"{obs_name}:joint"
                    )[:3]
                )

            # Record the initial object and obstacle positions
            self.init_object_pos: NDArray[np.float64] = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )[:3]
            self.init_obstacle_pos = init_obs_pos_list
            self.cumulative_obstacle_displacement = np.zeros(3)
            self.step_obstacle_displacement = np.zeros(3)
            self.prev_obstacle_pos: list[NDArray[np.float64]] = self.init_obstacle_pos

        self._mujoco.mj_forward(self.model, self.data)
        return True
