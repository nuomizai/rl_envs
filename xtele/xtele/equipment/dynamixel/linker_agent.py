from abc import abstractmethod
from typing import Optional, Sequence

import numpy as np

from xtele.equipment.dynamixel.driver.dynamixel_driver import DynamixelDriver
from xtele.equipment.dynamixel.driver.dynamixel_robot import DynamixelRobot
from xtele.equipment.tele_base import TeleBase


class LinkerBase(TeleBase):
    def __init__(self, identifier: str = ""):
        super().__init__(identifier)
        self._robot: Optional[DynamixelRobot] = None

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def act_dict(self):
        pass

    def set_position(self, target_position):
        self._robot.set_position(target_position)

    def set_position_torque(
        self,
        dxl_ids: Sequence[int],
        goal_positions: Sequence[float],
        goal_torques: Sequence[float],
        signs: Sequence[int],
    ):
        self._robot.set_position_torque(dxl_ids, goal_positions, goal_torques, signs)

    def act_vel(self):
        return self._robot.get_dxl_velocity()

    def act_freq(self):
        return self._robot.get_frequency()

    def deactivate_torque(self):
        self._robot.disable_torque()

    def activate_torque(self):
        self._robot.enable_torque()

    def switch_grav_mode(self):
        self._robot.switch_grav_mode()

    def switch_fb_mode(self):
        self._robot.switch_fb_mode()

    def close(self):
        self._robot.close()


class LinkerAgent(LinkerBase):
    """
    单臂Agent
    """

    def __init__(
        self,
        linker_config: dict,
        dynamic_config: Optional[dict] = None,
        identifier: str = "",
        driver: Optional[DynamixelDriver] = None,
    ):
        super().__init__(identifier)
        if "gripper" in linker_config.keys() and linker_config["gripper"]:
            joint_ids = linker_config["joint_ids"][:-1]
        else:
            joint_ids = linker_config["joint_ids"]
        self._robot = DynamixelRobot(
            joint_ids=joint_ids,
            joint_offsets=linker_config["joint_offsets"],
            joint_signs=linker_config["joint_signs"],
            port=linker_config["port"],
            driver=driver,
            dynamic_config=dynamic_config,
        )

        if "joint_limits" in linker_config.keys() and linker_config["joint_limits"]:
            self.joints_limits = np.deg2rad(linker_config["joint_limits"])
        else:
            self.joints_limits = None

    def act(self) -> np.ndarray:
        joints = self._robot.get_joints()
        if self.joints_limits is not None:
            joints = np.clip(joints, self.joints_limits[:, 0], self.joints_limits[:, 1])
        return joints

    def act_dict(self):
        joints = self.act()
        vels = self.act_vel()
        out_dict = {
            f"{self.identifier}_{i}": float(joint) for i, joint in enumerate(joints)
        }
        vel_dict = {
            f"{self.identifier}_{i}_vel": float(vel) for i, vel in enumerate(vels)
        }
        out_dict.update(vel_dict)

        freq, pos_update_time, vel_update_time = self.act_freq()
        out_dict[f"{self.identifier}_freq"] = freq
        out_dict[f"{self.identifier}_pos_upd_ts"] = pos_update_time
        out_dict[f"{self.identifier}_vel_upd_ts"] = vel_update_time

        return out_dict


class GripperAgent(LinkerBase):
    """
    Gripper Agent
    """

    def __init__(
        self,
        linker_config: dict,
        identifier: str = "",
        driver: Optional[DynamixelDriver] = None,
    ):
        super().__init__(identifier)
        assert "gripper" in linker_config.keys(), (
            "linker_config must have gripper config"
        )
        assert linker_config["gripper"] and len(linker_config["gripper"]) == 2, (
            "gripper config illegal"
        )
        self._robot = DynamixelRobot(
            joint_ids=linker_config["joint_ids"][-1:],
            joint_offsets=[0.0],
            joint_signs=[1],
            gripper_config=linker_config["gripper"],
            port=linker_config["port"],
            driver=driver,
        )

        if "gripper_limits" in linker_config.keys() and linker_config["gripper_limits"]:
            self.gripper_limits = np.asarray(linker_config["gripper_limits"])
        else:
            self.gripper_limits = None

    def act(self) -> np.ndarray:
        gripper = self._robot.get_joints()
        if self.gripper_limits is not None:
            return np.clip(gripper, self.gripper_limits[0], self.gripper_limits[1])
        else:
            return gripper

    def act_dict(self):
        gripper = self.act()
        return {f"{self.identifier}": float(gripper)}


class HeadLinkAgent(LinkerBase):
    """
    Head agent.
    """

    def __init__(
        self,
        linker_config: dict,
        identifier: str = "head",
        driver=None,
    ):
        super().__init__(identifier)
        self.joint_ids = linker_config["joint_ids"]
        self.joint_offsets = eval(linker_config["joint_offsets"])
        self._robot = DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=self.joint_offsets,
            joint_signs=linker_config["joint_signs"],
            port=linker_config["port"],
            driver=driver,
        )

        if "joint_limits" in linker_config.keys() and linker_config["joint_limits"]:
            self.joints_limits = np.deg2rad(linker_config["joint_limits"])
        else:
            self.joints_limits = None

        # init position
        self.reset_to_zero()

    def act(self) -> np.ndarray:
        joints = self._robot.get_joints()
        if self.joints_limits is not None:
            joints = np.clip(joints, self.joints_limits[:, 0], self.joints_limits[:, 1])
        return joints

    def act_dict(self):
        return {
            f"{self.identifier}_{i}": float(joint) for i, joint in enumerate(self.act())
        }

    def reset_to_zero(self):
        self._robot.set_position(self.joint_offsets)
        self._robot.disable_torque()

    def set_head_position(self, target_position):
        if set(target_position.keys()) == set(self.joint_ids):
            target_position = [target_position[dxl_id] for dxl_id in self.joint_offsets]
            self._robot.set_position(target_position)
        else:
            raise ValueError("Target position length error")


class WaistAgent(LinkerBase):
    """
    Waist Agent
    """

    def __init__(
        self,
        linker_config: dict,
        identifier: str = "waist",
        driver: Optional[DynamixelDriver] = None,
    ):
        super().__init__(identifier)
        self._robot = DynamixelRobot(
            joint_ids=linker_config["joint_ids"],
            joint_offsets=[eval(linker_config["joint_offsets"])],
            joint_signs=linker_config["joint_signs"],
            port=linker_config["port"],
            driver=driver,
        )

        if "joint_limits" in linker_config.keys() and linker_config["joint_limits"]:
            self.joints_limits = np.deg2rad(linker_config["joint_limits"])
        else:
            self.joints_limits = None

    def act(self) -> np.ndarray:
        joints = self._robot.get_joints()
        if self.joints_limits is not None:
            joints = np.clip(joints, self.joints_limits[:, 0], self.joints_limits[:, 1])
        return joints

    def act_dict(self):
        return {
            f"{self.identifier}": float(self.act()),
            f"{self.identifier}_vel": float(self.act_vel()),
        }


class MockLinkerAgent(LinkerBase):
    """
    测试Agent
    """

    def __init__(self, identifier, dof):
        super().__init__(identifier)
        self.dof = dof
        self.joints = np.zeros(dof)
        self.joints_vel = np.zeros(dof)

    def act(self) -> np.ndarray:
        return self.joints

    def act_dict(self):
        joints = self.act()
        vels = self.act_vel()
        out_dict = {
            f"{self.identifier}_{i}": float(joint) for i, joint in enumerate(joints)
        }
        vel_dict = {
            f"{self.identifier}_{i}_vel": float(vel) for i, vel in enumerate(vels)
        }
        out_dict.update(vel_dict)
        out_dict[f"{self.identifier}_freq"] = self.act_freq()
        return out_dict

    def act_vel(self):
        return self.joints_vel

    def act_freq(self):
        return 0

    def close(self):
        pass
