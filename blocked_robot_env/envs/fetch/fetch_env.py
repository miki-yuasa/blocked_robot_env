import numpy as np
from numpy.typing import NDArray

from gymnasium.utils import EzPickle
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv
from gymnasium_robotics.utils import rotations


class MujocoBlockedFetchEnv(MujocoFetchEnv):
    def __init__(self, reward_type="dense", **kwargs):
        super().__init__(reward_type, **kwargs)

    def _get_obs(self) -> dict[str, NDArray[np.float64]]:
        (
            grip_pos,
            object_pos,
            block_pos,
            object_rel_pos,
            block_rel_pos,
            gripper_state,
            object_rot,
            block_rot,
            object_velp,
            object_velr,
            block_velp,
            block_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                block_pos.ravel(),
                object_rel_pos.ravel(),
                block_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                block_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                block_velp.ravel(),
                block_velr.ravel(),
                grip_velp.ravel(),
                gripper_vel.ravel(),
            ]
        )

        achieved_goal = np.squeeze(object_pos.copy())

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

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

            # Add the block position, rotation, velocity, and angular velocity to the observation (from "object1")
            block_pos: NDArray[np.float64] = self.sim.data.get_site_xpos("object1")
            block_rot: NDArray[np.float64] = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object1")
            )
            block_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object1") * dt
            )
            block_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object1") * dt
            )
            # gripper state
            block_rel_pos = block_pos - grip_pos
            block_velp -= grip_velp
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
            block_pos,
            object_rel_pos,
            block_rel_pos,
            gripper_state,
            object_rot,
            block_rot,
            object_velp,
            object_velr,
            block_velp,
            block_velr,
            grip_velp,
            gripper_vel,
        )

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
