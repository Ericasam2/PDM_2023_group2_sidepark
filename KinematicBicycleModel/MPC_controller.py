import numpy as np
import sys

import importlib.util
import do_mpc
# print(do_mpc.__version__)
from kinetic_bicycle_model import KinematicBicycleModel
import matplotlib.pyplot as plt
from libs import CarDescription, StanleyController, generate_cubic_spline
from math import radians, pi
from casadi import cos, sin, tan, atan, acos, fmod, sqrt, hypot
from matplotlib import rcParams
from libs import normalise_angle
from csv import reader
from sklearn.metrics import mean_squared_error


class Path:

    def __init__(self):

        # Get path to waypoints.csv
        # data_path = 'data/sine_wave_waypoints.csv'
        data_path = 'data/waypoints.csv'
        with open(data_path, newline='') as f:
            rows = list(reader(f, delimiter=','))

        ds = 0.05
        x, y = [[float(i) for i in row] for row in zip(*rows[1:])]
        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)
        
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
        self.max_steer = radians(33)
        self.max_velocity = 5
        wheelbase = 2.96

        # Acceleration parameters
        
        self.target_velocity = 5.0
        self.time_to_reach_target_velocity = 5.0
        self.required_acceleration = self.target_velocity / self.time_to_reach_target_velocity

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
        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, wheelbase, self.px, self.py, self.pyaw)
        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, rear_length, self.max_steer, self.delta_time)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)
    
    def get_required_acceleration(self):

        self.time += self.delta_time
        return self.required_acceleration
    

    def plot_car(self):
        
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)

    def find_nearest_path_id(self, x, y, yaw):  

        # Calculate position of the front axle

        dx = x - self.px    # Find the x-axis of the front axle relative to the path
        dy = y - self.py    # Find the y-axis of the front axle relative to the path
        

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        target_index = np.argmin(d) # Find the shortest distance in the array
        
        return target_index, dx[target_index], dy[target_index], d[target_index]
    
    def calculate_crosstrack_term(self, yaw, dx, dy, absolute_error):
        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = np.sign(nearest_path_vector@front_axle_vector) * absolute_error
        return crosstrack_error
    
    def calculate_yaw_term(self, target_index, yaw):
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
        return yaw_error

    def drive(self):
        
        acceleration = 0 if self.time > self.time_to_reach_target_velocity else self.get_required_acceleration()
        if (sqrt((self.x - self.px[-1])**2 + (self.x - self.px[-1])**2) <= 6):
            if self.velocity > 0:
                acceleration = -2 * self.required_acceleration
            else:
                acceleration = 0
        print(self.time)
        self.wheel_angle, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.velocity, self.wheel_angle)
        print(self.target_id)
        self.x, self.y, self.yaw, self.velocity = self.kinematic_bicycle_model.dt_update(self.x, self.y, self.yaw, self.velocity, acceleration, self.wheel_angle)

        print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")
        

class MPC_controller:
    
    def __init__(self, vehicle, path_x, path_y, path_yaw):
        self.model_type = 'discrete'
        self.model = do_mpc.model.Model(self.model_type)
        
        self.px = path_x
        self.py = path_y
        self.pyaw = path_yaw
        self.vehicle = vehicle
        self.terminate = 0
        self.direction = -1
        self.goal_number = 0
        
        # error
        self.distance_error = []
        self.heading_error = []
    

    def mpc_setting(self):
        # Set parameters for MPC
        self.etup_mpc = {
            'n_horizon': 50,
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
    
    def mpc_dynamics(self):
        self.pos_x = self.model.set_variable('_x', 'position_x')
        self.pos_y = self.model.set_variable('_x', 'position_y')
        self.yaw = self.model.set_variable('_x', 'yaw')
        self.vel = self.model.set_variable('_x', 'velocity')

        self.a = self.model.set_variable('_u', 'acceleration')
        self.delta = self.model.set_variable('_u', 'steering')

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
        
        # define the reference signal as time-varing signals
        self.target_x = self.model.set_variable(var_type='_tvp', var_name='target_x')
        self.target_y = self.model.set_variable(var_type='_tvp', var_name='target_y')
        self.target_yaw = self.model.set_variable(var_type='_tvp', var_name='target_yaw')
        self.target_v = self.model.set_variable(var_type='_tvp', var_name='target_v')
        
    def find_target_path_id(self, x, y, yaw, goal_idx):  
        
        nearest_index, _, _, _ = self.find_nearest_path_id(x,y,yaw,goal_idx)
        # Calculate position of the front axle
        fx = x + self.direction * self.vehicle.wheelbase * cos(yaw)
        fy = y + self.direction * self.vehicle.wheelbase * sin(yaw)

        dx = fx - self.px    # Find the x-axis of the front axle relative to the path
        dy = fy - self.py    # Find the y-axis of the front axle relative to the path
        

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        print("goal index {}".format(goal_idx))
        if len(d[nearest_index:goal_idx]) <= 50:
            target_index = goal_idx-1
            self.goal_number += 1
        else:
            target_index = nearest_index + 10 + np.argmin(d[nearest_index + 10:goal_idx]) # Find the shortest distance in the array
        print("length: {}".format(len(d[nearest_index:goal_idx])))
        print(nearest_index)
        print(target_index)
        
        # yaw_error = normalise_angle(self.pyaw[target_index] - yaw)

        return target_index, dx[target_index], dy[target_index], d[target_index]
    
    def find_nearest_path_id(self, x, y, yaw, goal_idx):  

        # Calculate position of the front axle

        dx = x - self.px    # Find the x-axis of the front axle relative to the path
        dy = y - self.py    # Find the y-axis of the front axle relative to the path
        

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        target_index = np.argmin(d[:goal_idx]) # Find the shortest distance in the array
        
        return target_index, dx[target_index], dy[target_index], d[target_index]
    
    def calculate_crosstrack_term(self, yaw, dx, dy, absolute_error):
        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = np.sign(nearest_path_vector@front_axle_vector) * absolute_error
        return crosstrack_error
    
    def error_function(self):
        # define the error function
        # define according to the application
        x_error = (self.model.x["position_x"] - self.model.tvp["target_x"])**2
        y_error = (self.model.x["position_y"] - self.model.tvp["target_y"])**2
        yaw_error = (fmod((self.model.x["yaw"] - self.model.tvp["target_yaw"]) + 101*pi, 2*pi) - pi)**2
        vel_error = (self.model.x["velocity"] - self.model.tvp["target_v"])**2
        # x_error = (current_state[0] - self.target_state[0])**2
        # y_error = (current_state[1] - self.target_state[1])**2
        # yaw_error = (fmod((current_state[2] - self.target_state[2]) + 101*pi, 2*pi) - pi)**2
        # vel_error = (current_state[3] - self.target_state[3])**2
        # yaw_error = (sin(current_state[2] - self.target_state[2]))**2
        # print("current state: {}, {}, {}, {}".format(current_state[0], current_state[1], current_state[2], current_state[3]))
        # print("error: {}".format(x_error + y_error + yaw_error + vel_error))
        return 10*x_error + 10*y_error + 50*yaw_error + 10 * vel_error 
        # return 30*x_error + 1*y_error + 500*yaw_error + 1* vel_error 
    
    def terminal_error_function(self):
        # define the error function
        # define according to the application
        x_error = (self.model.x["position_x"] - self.px[-1])**2
        y_error = (self.model.x["position_y"] - self.py[-1])**2
        yaw_error = (fmod((self.model.x["yaw"] - self.pyaw[-1]) + 101*pi, 2*pi) - pi)**2
        vel_error = (self.model.x["velocity"] - 0)**2
        return 10*x_error + 10*y_error + 10*yaw_error + 100 * vel_error 
    
    def model_setup(self):
        
        # mpc setting
        self.mpc_setting()
        
        # dynamics
        self.mpc_dynamics()

        self.model.set_expression('cost', self.error_function())
        # self.model.set_expression('terminal cost', self.terminal_error_function())
         
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
        mpc.bounds['lower','_x','velocity'] =  -self.vehicle.max_velocity
        mpc.bounds['upper','_x','velocity'] =  self.vehicle.max_velocity
        '''
        nvp 
        '''
        mpc_tvp_template = mpc.get_tvp_template()
        # print(mpc_tvp_template)
        def mpc_tvp_fun(t):
            for k in range(50+1):
                mpc_tvp_template['_tvp', k, 'target_x'] = self.target_state[0]
                mpc_tvp_template['_tvp', k, 'target_y'] = self.target_state[1]
                mpc_tvp_template['_tvp', k, 'target_yaw'] = self.target_state[2]
                mpc_tvp_template['_tvp', k, 'target_v'] = self.target_state[3]
            return mpc_tvp_template
        mpc.set_tvp_fun(mpc_tvp_fun)
        # set up
        mpc.setup()
        estimator = do_mpc.estimator.StateFeedback(self.model)
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step = self.vehicle.delta_time)
        # Get the template
        stil_tvp_template = simulator.get_tvp_template()
        def stil_tvp_fun(t):
            stil_tvp_template['target_x'] = self.target_state[0]
            stil_tvp_template['target_y'] = self.target_state[1]
            stil_tvp_template['target_yaw'] = self.target_state[2]
            stil_tvp_template['target_v'] = self.target_state[3]
            # print(stil_tvp_template)

            return stil_tvp_template
        # Set the tvp_fun:
        simulator.set_tvp_fun(stil_tvp_fun)
        simulator.setup()
        
        return self.model, mpc, estimator, simulator
    
    def calculate_crosstrack_term(self, yaw, dx, dy, absolute_error):
        front_axle_vector = np.array([sin(yaw), -cos(yaw)])
        nearest_path_vector = np.array([dx, dy])
        crosstrack_error = np.sign(nearest_path_vector@front_axle_vector) * absolute_error
        return crosstrack_error
    
    def calculate_yaw_term(self, target_index, yaw):
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
        return yaw_error
    
    # cost function
    def update_reference(self, x, y, yaw, v, time, goals):
        # print(x)
        # print(y)
        # print(yaw)
        
        # find the target point
        print("goal number: {}".format(self.goal_number))
        mid_target_idx = int(goals[self.goal_number, 0])
        target_index, _, _, _ = self.find_target_path_id(x, y, yaw, int(goals[self.goal_number, 0]))
        target_x = self.px[target_index]
        target_y = self.py[target_index]
        target_yaw = self.pyaw[target_index]
        print("target yaw : {}".format(target_yaw))
        
        # error
        nearest_index, dx, dy, absolute_error = self.find_nearest_path_id(x, y, yaw, int(goals[self.goal_number, 0]))
        crosstrack_error = self.calculate_crosstrack_term(yaw, dx, dy, absolute_error)
        yaw_error = self.calculate_yaw_term(nearest_index, yaw)
        self.distance_error.append(crosstrack_error)
        self.heading_error.append(yaw_error)
        
        if self.goal_number == 0: self.direction = 1
        if self.goal_number == 1: self.direction = -1
        if self.goal_number == 2: self.direction = 1
        # determine the velocity control
        if self.direction == -1:
            target_yaw = fmod((target_yaw) + 3*pi, 2*pi) - pi
            target_v = 1/5 * self.direction * min(5, abs(float(x) - self.px[mid_target_idx-1]) + abs(float(y) - self.py[mid_target_idx-1]))
        else:
            target_v = self.direction * min(5, abs(float(x) - self.px[mid_target_idx-1]) + abs(float(y) - self.py[mid_target_idx-1]))
        # target_v = min(5, abs(float(x) - self.px[-1]) + abs(float(y) - self.py[-1]))
        print("target v: {}".format(target_v))
        self.target_state = np.array([target_x, target_y, target_yaw, target_v])
        

def MPC_drive_and_plot():
    X = []
    Y = []
    YAW = []
    V = []
    target_V = []
    target_X = []
    target_Y = []
    target_YAW = []
    path = Path()
    goal_idx = 4000
    car  = Car(path.px[0], path.py[0], path.pyaw[0], path.px[:goal_idx], path.py[:goal_idx], path.pyaw[:goal_idx], 1/50)
    controller = MPC_controller(car.kinematic_bicycle_model, path.px[:goal_idx], path.py[:goal_idx], path.pyaw[:goal_idx])
    initial_state = np.array([path.px[0], path.py[0], path.pyaw[0], 0])
    current_state = initial_state
    np.random.seed(99)
    # saved states
    X.append(initial_state[0])
    Y.append(initial_state[1])
    YAW.append(initial_state[2])
    V.append(initial_state[3])
    
    controller.update_reference(current_state[0], current_state[1], current_state[2], current_state[3], 0)
    target_X.append(controller.target_state[0])
    target_Y.append(controller.target_state[1])
    target_YAW.append(controller.target_state[2])
    target_V.append(controller.target_state[3])
    # control
    [model, mpc, estimator, simulator] = controller.model_setup()
    # while not controller.terminate == 1: 
    x0 = current_state
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0  
    # Use initial state to set the initial guess.
    mpc.set_initial_guess()
    N = 100
    for t in range(1, N):
        # control horizon
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        current_state = x0
        controller.update_reference(current_state[0], current_state[1], current_state[2], current_state[3], car.time)
        car.time += car.delta_time
        # save states
        X.append(float(current_state[0]))
        Y.append(float(current_state[1]))
        YAW.append(float(current_state[2]))
        V.append(float(current_state[3]))
        target_V.append(float(controller.target_state[3]))
        target_X.append(float(controller.target_state[0]))
        target_Y.append(float(controller.target_state[1]))
        target_YAW.append(float(controller.target_state[2]))
        
        if controller.terminate == 1:
            N = t
            print("Car reach the terminal state!")
            break
        
        
    
    # rcParams['axes.grid'] = True
    # rcParams['font.size'] = 18

    # Create subplots
    fig, axs = plt.subplots(2,2, figsize=(12, 12))
    tspam = [i for i in range(N) ]

    # Subplot 1: (X,Y) and (path.px, path.py)
    axs[0,0].plot(X, Y, linewidth=2)
    axs[0,0].plot(path.px[:goal_idx], path.py[:goal_idx], "--", linewidth=2)
    axs[0,0].scatter(X[-1], Y[-1], color='blue', s=50)  # Mark the final state
    axs[0,0].set_title("Position control using MPC controller", fontsize=14)
    axs[0,0].set_xlabel("X", fontsize=12)
    axs[0,0].set_ylabel("Y", fontsize=12)
    axs[0,0].legend(["Car", "Path", "Final Position"], fontsize=10)

    # Subplot 2: Velocity (V) and Target Velocity (target_V)
    axs[0,1].plot(tspam, V, linewidth=2)
    axs[0,1].plot(tspam, target_V, "--", linewidth=2)
    axs[0,1].set_title("Velocity control using MPC controller", fontsize=14)
    axs[0,1].set_xlabel("Control steps", fontsize=12)
    axs[0,1].set_ylabel("Velocity", fontsize=12)
    axs[0,1].legend(["Car Velocity", "Target Velocity"], fontsize=10)

    # Subplot 3: YAW and path.pyaw
    axs[1,0].plot(tspam, YAW, linewidth=2)
    axs[1,0].scatter(tspam[-1], YAW[-1], color='blue', s=50)  # Mark the final state
    axs[1,0].set_title("Yaw control using MPC controller", fontsize=14)
    axs[1,0].set_xlabel("Control steps", fontsize=12)
    axs[1,0].set_ylabel("YAW", fontsize=12)
    axs[1,0].legend(["Car YAW", "Final YAW"], fontsize=10)
    
    # Subplot 4: YAW and path.pyaw
    axs[1,1].plot([i for i in range(goal_idx)], path.pyaw[:goal_idx], "--", color='orange',linewidth=2)
    axs[1,1].set_title("Yaw on path", fontsize=14)
    axs[1,1].set_xlabel("waypoint index", fontsize=12)
    axs[1,1].set_ylabel("path", fontsize=12)
    
    # Add big title
    fig.suptitle("The MPC performance on reference tracking sin-Wave", fontsize=16)

    # Improve layout
    fig.tight_layout()

    # Save the figure to a file (e.g., PNG format)
    # plt.savefig('mpc_performance_plot.png')
    
    # Create a 2x1 grid of subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot distance error on the first subplot
    axs[0].plot(range(len(controller.distance_error)), controller.distance_error, linewidth=2, label='Distance Error')
    axs[0].set_xlabel("Control steps")
    axs[0].set_ylabel("Distance Error [m]")
    err_distance = mean_squared_error(np.zeros(len(controller.distance_error)), controller.distance_error)
    final_x_err = np.abs(X[-1] - path.px[goal_idx])
    final_y_err = np.abs(Y[-1] - path.py[goal_idx])
    axs[0].text(0.95, 0.95, f'Distance Error RMSE: {err_distance:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)
    axs[0].text(0.95, 0.85, f'Final X_err: {final_x_err:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)
    axs[0].text(0.95, 0.75, f'Final Y_err: {final_y_err:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)

    # Plot heading error on the second subplot
    axs[1].plot(range(len(controller.heading_error)), controller.heading_error, linewidth=2, label='Heading Error')
    axs[1].set_xlabel("Control steps")
    axs[1].set_ylabel("Yaw Error [rad]")
    err_heading = mean_squared_error(np.zeros(len(controller.heading_error)), controller.heading_error)
    final_yaw_err = normalise_angle(path.pyaw[goal_idx] - YAW[-1])
    axs[1].text(0.95, 0.95, f'Yaw Error RMSE: {err_heading:.2f} [rad]', transform=axs[1].transAxes, ha='right', va='top', fontsize=10)
    axs[1].text(0.95, 0.85, f'Final Yaw_err: {final_yaw_err:.2f} [rad]', transform=axs[1].transAxes, ha='right', va='top', fontsize=10)

    # Set the title for the entire figure
    fig.suptitle("The tracking error of the MPC controller with sin-Wave path")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot to a file (e.g., PNG format)
    # plt.savefig('tracking_error_plot.png')

    # Show the plots
    plt.show()

def Stanley_drive_and_plot():
    X = []
    Y = []
    YAW = []
    V = []
    distance_error = []
    heading_error = []
    path = Path()
    goal_idx = 4000
    car  = Car(path.px[0], path.py[0], path.pyaw[0], path.px[:goal_idx], path.py[:goal_idx], path.pyaw[:goal_idx], 1/50)
    initial_state = np.array([path.px[0], path.py[0], path.pyaw[0], 0])
    current_state = initial_state
    # saved states
    X.append(initial_state[0])
    Y.append(initial_state[1])
    YAW.append(initial_state[2])
    V.append(initial_state[3])
    N = 2300
    for t in range(1, N):
        car.drive()
        current_state = np.array([car.x, car.y, car.yaw, car.velocity])
        X.append(float(current_state[0]))
        Y.append(float(current_state[1]))
        YAW.append(float(current_state[2]))
        V.append(float(current_state[3]))
        
        nearest_index, dx, dy, absolute_error = car.find_nearest_path_id(car.x, car.y, car.yaw)
        crosstrack_error = car.calculate_crosstrack_term(car.yaw, dx, dy, absolute_error)
        yaw_error = car.calculate_yaw_term(nearest_index, car.yaw)
        distance_error.append(crosstrack_error)
        heading_error.append(yaw_error)
        
    # Create subplots
    fig, axs = plt.subplots(2,2, figsize=(12, 12))
    tspam = [i for i in range(N) ]

    # Subplot 1: (X,Y) and (path.px, path.py)
    axs[0,0].plot(X, Y, linewidth=2)
    axs[0,0].plot(path.px[:goal_idx], path.py[:goal_idx], "--", linewidth=2)
    axs[0,0].scatter(X[-1], Y[-1], color='blue', s=50)  # Mark the final state
    axs[0,0].set_title("Position control using MPC controller", fontsize=14)
    axs[0,0].set_xlabel("X", fontsize=12)
    axs[0,0].set_ylabel("Y", fontsize=12)
    axs[0,0].legend(["Car", "Path", "Final Position"], fontsize=10)

    # Subplot 3: YAW and path.pyaw
    axs[1,0].plot(tspam, YAW, linewidth=2)
    axs[1,0].scatter(tspam[-1], YAW[-1], color='blue', s=50)  # Mark the final state
    axs[1,0].set_title("Yaw control using MPC controller", fontsize=14)
    axs[1,0].set_xlabel("Control steps", fontsize=12)
    axs[1,0].set_ylabel("YAW", fontsize=12)
    axs[1,0].legend(["Car YAW", "Final YAW"], fontsize=10)
    
    # Subplot 4: YAW and path.pyaw
    axs[1,1].plot([i for i in range(goal_idx)], path.pyaw[:goal_idx], "--", color='orange',linewidth=2)
    axs[1,1].set_title("Yaw on path", fontsize=14)
    axs[1,1].set_xlabel("waypoint index", fontsize=12)
    axs[1,1].set_ylabel("path", fontsize=12)
    
    # Add big title
    fig.suptitle("The Stanley Controller performance on reference tracking sin-Wave", fontsize=16)

    # Improve layout
    fig.tight_layout()
    
    # Create a 2x1 grid of subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot distance error on the first subplot
    axs[0].plot(range(len(distance_error)), distance_error, linewidth=2, label='Distance Error')
    axs[0].set_xlabel("Control steps")
    axs[0].set_ylabel("Distance Error [m]")
    err_distance = mean_squared_error(np.zeros(len(distance_error)), distance_error)
    final_x_err = np.abs(X[-1] - path.px[goal_idx])
    final_y_err = np.abs(Y[-1] - path.py[goal_idx])
    axs[0].text(0.95, 0.95, f'Distance Error RMSE: {err_distance:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)
    axs[0].text(0.95, 0.85, f'Final X_err: {final_x_err:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)
    axs[0].text(0.95, 0.75, f'Final Y_err: {final_y_err:.2f} [m]', transform=axs[0].transAxes, ha='right', va='top', fontsize=10)

    # Plot heading error on the second subplot
    axs[1].plot(range(len(heading_error)), heading_error, linewidth=2, label='Heading Error')
    axs[1].set_xlabel("Control steps")
    axs[1].set_ylabel("Yaw Error [rad]")
    err_heading = mean_squared_error(np.zeros(len(heading_error)), heading_error)
    final_yaw_err = normalise_angle(path.pyaw[goal_idx] - YAW[-1])
    axs[1].text(0.95, 0.95, f'Yaw Error RMSE: {err_heading:.2f} [rad]', transform=axs[1].transAxes, ha='right', va='top', fontsize=10)
    axs[1].text(0.95, 0.85, f'Final Yaw_err: {final_yaw_err:.2f} [rad]', transform=axs[1].transAxes, ha='right', va='top', fontsize=10)

    # Set the title for the entire figure
    fig.suptitle("The tracking error of the Stanley controller with sin-Wave path")

    # Adjust layout for better spacing
    plt.tight_layout()
    
    plt.savefig('tracking_error_plot_stanley.png')
    plt.show()
    
def main():
    MPC_drive_and_plot()
    # Stanley_drive_and_plot()
        
    
if __name__ == '__main__':
    main()
