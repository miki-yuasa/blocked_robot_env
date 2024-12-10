from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray

from gymnasium_robotics.envs.fetch.fetch_env import goal_distance, DEFAULT_CAMERA_CONFIG
from gymnasium_robotics.utils import rotations

from safety_robot_env.envs.robot_env import MujocoRobotEnv


class MujocoBlockedFetchEnv(MujocoRobotEnv):
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
        distance_threshold: float,
        reward_type: Literal["sparse", "dense"],
        goal_reward: float = 10.0,
        obstacle_penalty: bool = True,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
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
        self.distance_threshold: float = distance_threshold
        self.reward_type: Literal["sparse", "dense"] = reward_type
        self.goal_reward: float = goal_reward
        self.obstacle_penalty: bool = obstacle_penalty

        self.init_obstacle_pos: NDArray[np.float64] = np.zeros(3)
        self.cumulative_obstacle_displacement: NDArray[np.float64] = np.zeros(3)

        if self.reward_type not in ["sparse", "dense"]:
            raise ValueError("Invalid reward type. Must be either 'sparse' or 'dense'.")
        else:
            pass

        super().__init__(
            n_actions=4, default_camera_config=default_camera_config, **kwargs
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float64],
        goal: NDArray[np.float64],
        info: dict[str, Any] | list[dict[str, Any]],
    ):
        # Compute distance between goal and the achieved goal.
        d: NDArray[np.float64] = goal_distance(achieved_goal, goal)

        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            reward: NDArray[np.float64] = -d
            reward += self.goal_reward * (d < self.distance_threshold).astype(
                np.float64
            )
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

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

        return action

    def _get_obs(self) -> dict[str, NDArray[np.float64]]:
        (
            grip_pos,
            object_pos,
            obstacle_pos,
            object_rel_pos,
            obstacle_rel_pos,
            gripper_state,
            object_rot,
            obstacle_rot,
            object_velp,
            object_velr,
            obstacle_velp,
            obstacle_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                obstacle_pos.ravel(),
                object_rel_pos.ravel(),
                obstacle_rel_pos.ravel(),
                # self.init_obstacle_pos.ravel(),
                # self.cumulative_obstacle_displacement.ravel(),
                gripper_state,
                object_rot.ravel(),
                obstacle_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                obstacle_velp.ravel(),
                obstacle_velr.ravel(),
                grip_velp.ravel(),
                gripper_vel.ravel(),
            ]
        )

        achieved_goal = np.concatenate(
            [
                object_pos.copy(),
                (
                    self.cumulative_obstacle_displacement.copy()
                    if self.obstacle_penalty
                    else np.zeros(3)
                ),
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _get_dict_obs(self) -> dict[str, NDArray[np.float64]]:
        (
            grip_pos,
            object_pos,
            obstacle_pos,
            object_rel_pos,
            obstacle_rel_pos,
            gripper_state,
            object_rot,
            obstacle_rot,
            object_velp,
            object_velr,
            obstacle_velp,
            obstacle_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        dict_obs: dict[str, NDArray[np.float64]] = {
            "grip_pos": grip_pos,
            "object_pos": object_pos,
            "obstacle_pos": obstacle_pos,
            "object_rel_pos": object_rel_pos,
            "obstacle_rel_pos": obstacle_rel_pos,
            "gripper_state": gripper_state,
            "object_rot": object_rot,
            "obstacle_rot": obstacle_rot,
            "object_velp": object_velp,
            "object_velr": object_velr,
            "obstacle_velp": obstacle_velp,
            "obstacle_velr": obstacle_velr,
            "grip_velp": grip_velp,
            "gripper_vel": gripper_vel,
        }

        return dict_obs

    def _get_info(self) -> dict[str, Any]:
        info: dict[str, NDArray[np.float64]] = self._get_dict_obs()
        normal_obs: dict[str, NDArray[np.float64]] = self._get_obs()
        info["is_success"] = self._is_success(normal_obs["achieved_goal"], self.goal)
        info["init_object_pos"] = self.init_object_pos
        info["init_obstacle_pos"] = self.init_obstacle_pos
        info["goal"] = self.goal

        return info

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
        # Extend the goal to include cumulative displacement of the obstacle to the goal in the x, y, and z directions
        goal = np.concatenate([goal, np.zeros(3)])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _step_callback(self):

        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

        # Record the displacement of the obstacle
        curr_obstacle_pos: NDArray[np.float64] = self._utils.get_site_xpos(
            self.model, self.data, "object1"
        )
        step_obstacle_displacement: NDArray[np.float64] = (
            curr_obstacle_pos - self.prev_obstacle_pos
        )
        self.cumulative_obstacle_displacement += step_obstacle_displacement
        self.prev_obstacle_pos = curr_obstacle_pos

    def generate_mujoco_observations(self) -> tuple[NDArray[np.float64], ...]:
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

            # Add the obstacle position, rotation, velocity, and angular velocity to the observation (from "object1")
            obstacle_pos: NDArray[np.float64] = self._utils.get_site_xpos(
                self.model, self.data, "object1"
            )
            obstacle_rot: NDArray[np.float64] = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object1")
            )
            obstacle_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object1") * dt
            )
            obstacle_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object1") * dt
            )
            # gripper state
            obstacle_rel_pos = obstacle_pos - grip_pos
            obstacle_velp -= grip_velp
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
            obstacle_pos,
            object_rel_pos,
            obstacle_rel_pos,
            gripper_state,
            object_rot,
            obstacle_rot,
            object_velp,
            object_velr,
            obstacle_velp,
            obstacle_velr,
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

            # Randomize start position of obstacle.
            obstacle_xpos = self.initial_gripper_xpos[:2]
            # Ensure obstacle is not placed on top of object and is not too close to the gripper
            while (
                np.linalg.norm(obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1
                or np.linalg.norm(obstacle_xpos - object_xpos) < 0.1
            ):
                obstacle_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            obstacle_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object1:joint"
            )
            assert obstacle_qpos.shape == (7,)
            obstacle_qpos[:2] = obstacle_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object1:joint", obstacle_qpos
            )

            # Record the initial object and obstacle positions
            self.init_object_pos: NDArray[np.float64] = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )[:3]
            self.init_obstacle_pos = self._utils.get_joint_qpos(
                self.model, self.data, "object1:joint"
            )[:3]
            self.cumulative_obstacle_displacement = np.zeros(3)
            self.prev_obstacle_pos: NDArray[np.float64] = self.init_obstacle_pos

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos: dict[str, NDArray[np.float64]]) -> None:
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
