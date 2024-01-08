import numpy as np
import sys

import importlib.util
import do_mpc
# print(do_mpc.__version__)
from kinetic_bicycle_model import KinematicBicycleModel
import matplotlib.pyplot as plt
from libs import CarDescription, StanleyController, generate_cubic_spline
from math import radians, cos, sin, atan, tan

from matplotlib import rcParams

class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, delta_time):

        # Model parameters
        self.x = init_x  # x coordinate
        self.y = init_y  # y coordinate
        self.yaw = init_yaw  # orientation
        self.delta_time = delta_time  # sampling time
        self.time = 0.0
        self.velocity = 0.0
        self.wheel_angle = 0.0
        self.angular_velocity = 0.0
        rear_length = 2.0
        max_steer = radians(33)
        wheelbase = 2.96

        # Acceleration parameters
        target_velocity = 10.0
        self.time_to_reach_target_velocity = 5.0
        self.required_acceleration = target_velocity / self.time_to_reach_target_velocity

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # Description parameters
        self.colour = 'black'
        overall_length = 4.97
        overall_width = 1.964
        tyre_diameter = 0.4826
        tyre_width = 0.265
        axle_track = 1.7
        rear_overhang = 0.5 * (overall_length - wheelbase)

        # Path tracking and Bicycle model
        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, max_steer, wheelbase, self.px, self.py, self.pyaw)
        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, rear_length, max_steer, self.delta_time)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)

    
    def get_required_acceleration(self):

        self.time += self.delta_time
        return self.required_acceleration
    

    def plot_car(self):
        
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)


    def drive(self):
        
        acceleration = 0 if self.time > self.time_to_reach_target_velocity else self.get_required_acceleration()
        self.wheel_angle, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.velocity, self.wheel_angle)
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(self.x, self.y, self.yaw, self.velocity, acceleration, self.wheel_angle)

        print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")
        

class MPC_controller:
    
    def __init__(self, vehicle, initial_state, target_state):
        self.model_type = 'discrete'
        self.model = do_mpc.model.Model(self.model_type)
        
        self.initial_state = initial_state
        self.target_state = target_state
    
        self.vel = self.model.set_variable('_x', 'velocity')
        self.pos_x = self.model.set_variable('_x', 'position_x')
        self.pos_y = self.model.set_variable('_x', 'position_y')
        self.yaw = self.model.set_variable('_x', 'yaw')

        self.a = self.model.set_variable('_u', 'acceleration')
        self.delta = self.model.set_variable('_u', 'steering')

        self.vehicle = vehicle
        [self.new_pos_x, self.new_pos_y, self.new_yaw, self.new_vel] = self.vehicle.dt_update(self.pos_x, 
                                                                                            self.pos_y, 
                                                                                            self.yaw, 
                                                                                            self.vel, 
                                                                                            self.a, 
                                                                                            self.delta)
        self.model.set_rhs('position_x', self.new_pos_x)
        self.model.set_rhs('position_y', self.new_pos_y)
        self.model.set_rhs('yaw', self.new_yaw)
        self.model.set_rhs('velocity', self.new_vel)
        self.model.set_expression('cost', self.error_function(np.array([self.pos_x, self.pos_y, self.yaw, self.vel])))
        # Set parameters for MPC
        self.etup_mpc = {
            'n_horizon': 100,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.vehicle.delta_time,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
    
    def error_function(self, current_state):
        # define the error function
        # define according to the application
        x_error = (current_state[0] - self.target_state[0])**2
        y_error = (current_state[1] - self.target_state[1])**2
        yaw_error = (current_state[2] - self.target_state[2])**2
        vel_error = (current_state[3] - self.target_state[3])**2
        # print("current state: {}, {}, {}, {}".format(current_state[0], current_state[1], current_state[2], current_state[3]))
        # print("error: {}".format(x_error + y_error + yaw_error + vel_error))
        return x_error + y_error + yaw_error + vel_error 
    
    
    # cost function
    def generate(self):
        
        # Build the model
        self.model.setup()

        # MPC controller
        mpc = do_mpc.controller.MPC(self.model)
        mpc.set_param(**self.etup_mpc)
        
        # state cost
        mterm = self.model.aux['cost'] # terminal cost
        lterm = self.model.aux['cost'] # stage cost
        mpc.set_objective(mterm=mterm, lterm=lterm)
        
        # input cost
        mpc.set_rterm(acceleration=1e-4) # input penalty
        mpc.set_rterm(steering=1e-4) # input penalty
        
        # constraints
        # lower bounds of the input
        mpc.bounds['lower','_u','steering'] = -self.vehicle.max_steer
        # upper bounds of the input
        mpc.bounds['upper','_u','steering'] =  self.vehicle.max_steer
        
        # set up
        mpc.setup()
        estimator = do_mpc.estimator.StateFeedback(self.model)
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step = self.vehicle.delta_time)
        simulator.setup()
        
        return self.model, mpc, estimator, simulator
        
        
        
        
    
def main():
    car  = Car(0, 0, 0, 50, 50, 3, 1/50.0)
    initial_state = np.array([0, 0, 0, 5])
    target_state = np.array([10, 10, 2, 5])
    controller = MPC_controller(car.kinematic_bicycle_model, initial_state, target_state)
    [model, mpc, estimator, simulator] = controller.generate()
    
    # simulation
    
    # Seed
    np.random.seed(99)

    # Initial state
    e = np.ones([model.n_x,1])
    x0 = initial_state
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0
    
    # Use initial state to set the initial guess.
    mpc.set_initial_guess()
    
    n_steps = 200
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
    
    
    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18
    
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.show()
    
    
if __name__ == '__main__':
    main()
