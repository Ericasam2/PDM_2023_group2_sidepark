from math import cos, sin, tan, atan

from libs import normalise_angle


class KinematicBicycleModel:
    """
    Summary
    -------
    This class implements the 2D Kinematic Bicycle Model for vehicle dynamics

    Attributes
    ----------
    dt (float) : discrete time period [s]
    wheelbase (float) : vehicle's wheelbase [m]
    max_steer (float) : vehicle's steering limits [rad]

    Methods
    -------
    __init__(wheelbase: float, max_steer: float, delta_time: float=0.05)
        initialises the class

    update(x, y, yaw, velocity, acceleration, steering_angle)
        updates the vehicle's state using the kinematic bicycle model
    """
    def __init__(self, wheelbase: float, rear_length: float, max_steer: float, delta_time: float=0.05):

        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.rear_length = rear_length


    def update(self, x: float, y: float, yaw: float, velocity: float, acceleration: float, steering_angle: float) -> tuple[float, ...]:
        """
        Summary
        -------
        Updates the vehicle's state using the kinematic bicycle model

        Parameters
        ----------
        x (int) : vehicle's x-coordinate [m]
        y (int) : vehicle's y-coordinate [m]
        yaw (int) : vehicle's heading [rad]
        velocity (int) : vehicle's velocity in the x-axis [m/s]
        acceleration (int) : vehicle's accleration [m/s^2]
        steering_angle (int) : vehicle's steering angle [rad]

        Returns
        -------
        new_x (int) : vehicle's x-coordinate [m]
        new_y (int) : vehicle's y-coordinate [m]
        new_yaw (int) : vehicle's heading [rad]
        new_velocity (int) : vehicle's velocity in the x-axis [m/s]
        steering_angle (int) : vehicle's steering angle [rad]
        angular_velocity (int) : vehicle's angular velocity [rad/s]
        """
        
        # compute velocity in vehicle longitudinal direction
        new_velocity = velocity + self.delta_time * acceleration
        # compute the steering angle (system input)
        steering_angle = -self.max_steer if steering_angle < -self.max_steer else self.max_steer if steering_angle > self.max_steer else steering_angle
        # compute vehicle slip angle  
        slip_angle = atan(self.rear_length * tan(steering_angle)) 
        # compute the vechile yaw, x, y
        yaw_dot = new_velocity * sin(slip_angle) / self.rear_length
        x_dot = new_velocity * cos(yaw + slip_angle)
        y_dot = new_velocity * sin(yaw + slip_angle)
        
        return x_dot, y_dot, yaw_dot, new_velocity, steering_angle
