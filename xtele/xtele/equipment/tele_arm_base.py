from abc import abstractmethod


class TeleArmBase:
    def __init__(self):
        pass

    ## 1- Must be implemented.
    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the tele-arm.

        Returns:
            int: The number of joints of the tele-arm.
        """
        raise NotImplementedError

    def get_frequency(self):
        pass

    def get_position(self):
        pass

    def get_velocity(self):
        pass

    def reach_home(self):
        pass

    def enable_torque(self):
        pass

    def disable_torque(self):
        pass

    def close(self):
        pass

    def reach_position(self):
        """
        With interpolate method
        """
        pass

    def sync_postition(self):
        """
        Without interpolate method
        """
        pass

    def set_torque(self):
        pass

    def activate_grav_mode(self):
        pass

    def activate_fb_mode(self):
        pass
