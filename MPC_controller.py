import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from math import radians
from KinematicBicycleModel.kinetic_bicycle_model import KinematicBicycleModel
from libs import CarDescription, StanleyController, generate_cubic_spline
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


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
        # vehicle states
        self.initial_state = initial_state
        self.vehicle = vehicle
        self.target_state = target_state
        # MPC
        self.input_horizon = 3
        self.horizon = 15
        self.constraints = []
        self.cost = 0
        # CVX parameters
        self.dim_state = 4
        self.dim_input = 1
    
    def error_function(self, current_state):
        # define the error function
        # define according to the application
        x_error = abs(current_state[0] - self.target_state[0])
        y_error = abs(current_state[1] - self.target_state[1])
        # yaw_error = abs(np.sin(current_state[2] - self.target_state[2]))
        yaw_error = 0
        vel_error = abs(current_state[3] - self.target_state[3])
        # print("current state: {}, {}, {}, {}".format(current_state[0], current_state[1], current_state[2], current_state[3]))
        # print("error: {}".format(x_error + y_error + yaw_error + vel_error))
        return x_error + y_error + yaw_error + vel_error
    
    def cost_function(self, variables):
        x = variables[:self.dim_state * (self.horizon + 1)].reshape((self.dim_state, self.horizon+1))
        u = variables[self.dim_state * (self.horizon + 1):].reshape((self.dim_input, self.horizon))
        
        cost = 0.0
        
        Q = 0.2 * np.eye(1)  # state weight matrix
        L = 0.1 * np.eye(self.dim_input)  # input weight matrix
        
        for k in range(self.horizon):
            '''
            L @ u need to be scaller
            '''
            cost += Q * self.error_function(x[:,k])**2 + u[:,k] @ L @ u[:,k]
            # print("cost: {}".format(cost))
        
        # print("cost: {}".format(cost))
        return cost
    
    def dynamics_constraint(self, variables, k):
        x = variables[:self.dim_state * (self.horizon + 1)].reshape((self.dim_state, self.horizon+1))
        u = variables[self.dim_state * (self.horizon + 1):].reshape((self.dim_input, self.horizon))
        return x[:,k+1] - self.vehicle.dt_update(x[0,k], x[1,k], x[2,k], 5, 0, u[0,k])
        
    def constraint_function(self, variables):
        x = variables[:self.dim_state * (self.horizon + 1)].reshape((self.dim_state, self.horizon+1))
        u = variables[self.dim_state * (self.horizon + 1):].reshape((self.dim_input, self.horizon))
        
        constraints = []
        
        for k in range(self.horizon):
            # constraints += [NonlinearConstraint(lambda variables: variables[:self.dim_state] - self.initial_state, 0, 0)]
            constraints += [{'type': 'ineq', 'fun': lambda x: u[:,k] - self.vehicle.max_steer}]
            constraints += [{'type': 'ineq', 'fun': lambda x: -u[:,k] - self.vehicle.max_steer}]
            # constraints += [{'type': 'ineq', 'fun': lambda t: x[2,k] - np.pi}]
            # constraints += [{'type': 'ineq', 'fun': lambda t: -x[2,k] - np.pi}]
            # print("x: {}".format(x))
            '''have some bug with dynamics'''
            constraints += [ NonlinearConstraint(lambda variables: self.dynamics_constraint(variables, k), -1e-4, 1e-4) ]
            # constraints += [{'type': 'eq', 'fun': lambda x: self.dynamics_constraint(variables, k)}]
        
        return constraints
    
    def onestep(self, current_state):
        # controller design
        # initial guess
        x0 = np.zeros((self.dim_state * (self.horizon+1) + self.dim_input * (self.horizon), ))
        
        # initial states
        constraints = self.constraint_function(x0)
        constraints += [NonlinearConstraint(lambda variables: variables[:self.dim_state] - current_state, -1e-4, 1e-4)]
        # print("constraints: \n{}".format(constraints))
        
        # print("constraints: \n{}".format(constraints))
        # print("we have {} constraints".format(len(constraints)))
        # Solves the problem
        options = {'maxiter': 200}  # Increase this number as needed
        result = minimize(self.cost_function, x0, method="SLSQP", constraints=constraints, options=options)
        
        opt_inputs = result.x[self.dim_state * (self.horizon + 1):].reshape((self.dim_input, self.horizon))
        opt_states = result.x[:self.dim_state * (self.horizon + 1)].reshape((self.dim_state, self.horizon+1))
        
        print("At state {}".format(current_state))
        
        if result.success:
            print("Optimizer find an optimal solution")
        else:
            print("Optimizer failed to find an optimal solution")
        print(result.message)

        # We return the MPC input and the next state
        return opt_inputs, opt_states

    def generate(self):
        # controller design        
        # initial states
        k = 0
        inputs = []
        current_state = self.initial_state
        while (k < 100):
            [U, _] = self.onestep(current_state)
            k += 1
            # pick the first one
            current_u = U.squeeze()[:self.input_horizon]
            inputs = np.array(list(inputs)+ list(current_u))
            print(inputs)
            # one step forward
            for i in range(self.input_horizon):
                new_state = self.vehicle.dt_update(current_state[0], current_state[1], current_state[2], 5, 0, current_u[i])
                current_state = new_state

        # We return the MPC input and the next state
        return inputs
    
def dynamics_test(car):
    X = np.zeros([101])
    Y = np.zeros([101])
    Yaw = np.zeros([101])
    inputs = 0.1* np.ones([100])
    inputs[50:] = inputs[50:] * 1
    for i in range(len(inputs)):
        [X[i+1], Y[i+1], Yaw[i+1], _, _] = car.kinematic_bicycle_model.update(X[i], Y[i], Yaw[i], 5, 0, inputs[i])
    Xd = np.zeros([101])
    Yd = np.zeros([101])
    Yawd = np.zeros([101])
    for i in range(len(inputs)):
        [Xd[i+1], Yd[i+1], Yawd[i+1], _] = car.kinematic_bicycle_model.dt_update(Xd[i], Yd[i], Yawd[i], 5, 0, inputs[i])
    plt.subplot(3,1,1)
    plt.plot(np.linspace(0, 101, 101), X)
    plt.plot(np.linspace(0, 101, 101), Y)
    plt.plot(np.linspace(0, 101, 101), Yaw)
    plt.legend(["X", "Y", "Yaw"])
    plt.xlabel("time")
    plt.ylabel("state")
    
    plt.subplot(3,1,2)
    plt.plot(np.linspace(0, 101, 101), Xd)
    plt.plot(np.linspace(0, 101, 101), Yd)
    plt.plot(np.linspace(0, 101, 101), Yawd)
    plt.legend(["Xd", "Yd", "Yawd"])
    plt.xlabel("time")
    plt.ylabel("state")
    
    plt.subplot(3,1,3)
    plt.plot(Xd, Yd)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    """_summary_
    1. when the input is 0, the car go straight line, the x keep increasing and other states remain constant
    2. when the input is >0, <3, the car is doing ackermaan steering, the states have a periodical behaviour
    3. when the input >3, the car's cannot keep doing ackermaan steering anymore, the state go unbounded.
    """
    
def simulation(car, inputs, initial_state, target_state, dim_state=4, dim_input=1):
    l = len(inputs) + 1
    states = np.zeros((dim_state, len(inputs)+1))
    states[:,0] = initial_state
    for k in range(len(inputs)):
        states[:,k+1] = car.kinematic_bicycle_model.dt_update(states[0, k], states[1, k], states[2, k], states[3, k], 0, inputs[k])
    plt.subplot(2,1,1)
    plt.plot(np.linspace(0, l, l), states[0,:])
    plt.plot(np.linspace(0, l, l), states[1,:])
    plt.plot(np.linspace(0, l, l), states[2,:])
    plt.plot(np.linspace(0, l, l), states[3,:])
    plt.legend(["X", "Y", "Yaw", "velocity"])
    plt.xlabel("time")
    plt.ylabel("state")
    
    plt.subplot(2,1,2)
    plt.plot(np.linspace(0, l-1, l-1), inputs[:])
    plt.legend(["Input"])
    plt.xlabel("time")
    plt.ylabel("inputs")
    plt.show()
    
    
    
def main():
    car  = Car(0, 0, 0, 50, 50, 3, 1/50.0)
    # dynamics_test(car)
    initial_state = np.array([0, 0, 0, 5])
    target_state = np.array([3, 3, 2, 5])
    controller = MPC_controller(car.kinematic_bicycle_model, initial_state, target_state)
    # [U] = controller.generate()
    # print(U.squeeze())
    # print(X.squeeze())
    # simulation(car, U.squeeze(), initial_state, target_state)
    # plt.plot(np.linspace(0, len(U), len(U)), U[0,:])
    # plt.plot(np.linspace(0, len(U), len(U)), U[1,:])
    # plt.legend(['steer'])
    dynamics_test(car)
    
    
if __name__ == '__main__':
    main()
