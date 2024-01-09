from math import pi
import numpy as np
import cvxpy as cp
from libs import normalise_angle
from casadi import cos, sin, tan, atan

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
        # compute vehicle slip angle  
        slip_angle = atan(self.rear_length * tan(steering_angle) / self.wheelbase) 
        # compute the vechile yaw, x, y
        yaw_dot = velocity * sin(slip_angle) / self.rear_length
        x_dot = velocity * cos(yaw + slip_angle)
        y_dot = velocity * sin(yaw + slip_angle)
        
        new_x = x + x_dot * self.delta_time
        new_y = y + y_dot * self.delta_time
        new_yaw = (yaw + yaw_dot * self.delta_time)
        if new_yaw > 2 * pi:
            new_yaw = new_yaw - 2*pi
        elif new_yaw < 0:
            new_yaw = new_yaw + 2*pi
        # new_yaw = (new_yaw + pi) % (2 * pi) - pi
        return new_x, new_y, new_yaw, new_velocity, steering_angle
    
    
    def dt_update(self, x: float, y: float, yaw: float, velocity: float, acceleration: float, steering_angle: float) -> tuple[float, ...]:
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
        slip_angle = atan(self.rear_length * tan(steering_angle) / self.wheelbase) 
        # compute the vechile yaw, x, y
        yaw_dot = velocity * sin(slip_angle) / self.rear_length
        x_dot = velocity * cos(yaw + slip_angle)
        y_dot = velocity * sin(yaw + slip_angle)
        
        new_x = x + x_dot * self.delta_time
        new_y = y + y_dot * self.delta_time
        new_yaw = yaw + yaw_dot * self.delta_time
        return new_x, new_y, new_yaw, new_velocity
    
    
    def linear_dt_dynamics(self, x: float, y: float, yaw: float, velocity: float, acceleration: float, steering_angle: float):
        steering_angle = steering_angle
        # compute vehicle slip angle  
        slip_angle = atan(self.rear_length * tan(steering_angle) / self.wheelbase)
        Ad = np.array([[1,0,0,0], 
                       [sin(yaw+slip_angle)*self.delta_time, 1, 0, -velocity*sin(yaw+slip_angle)*self.delta_time],
                       [cos(yaw+slip_angle)*self.delta_time, 0, 1, velocity*cos(yaw+slip_angle)*self.delta_time],
                       [sin(slip_angle)/self.rear_length*self.delta_time, 0, 0, 1]])
        bd = np.array([[self.delta_time, 0],
                       [0, -velocity*sin(yaw+slip_angle)*self.delta_time],
                       [0, velocity*cos(yaw+slip_angle)*self.delta_time],
                       [0, velocity/self.rear_length*cos(slip_angle)*self.delta_time]])
        return Ad, bd
    