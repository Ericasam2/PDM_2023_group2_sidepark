import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

class MPC_controller():
    def __init__(self, vehicle, measure_state, initial_state, target_state):
        # vehicle states
        self.initial_state = initial_state
        self.vehicle = vehicle
        self.measure_state = measure_state
        self.target_state = target_state
        # MPC
        self.input_horizon = 2
        self.horizon = 10
        self.constraints = []
        self.cost = 0
        # CVX parameters
        self.dim_state = 3
        self.dim_input = 1
    
    def error_function(self, measure_state):
        # define the error function
        # define according to the application
        x_error = abs(measure_state[0] - self.target_state[0])
        y_error = abs(measure_state[1] - self.target_state[1])
        yaw_error = abs(np.sin(measure_state[2] - self.target_state[2]))
        return x_error + y_error + yaw_error
    
    def controller(self):
        # controller design
        # cvx
        ## Create the optimization variables
        x = cp.Variable((self.dim_state, self.horizon + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((self.dim_input, self.horizon))
        # penalty matrix
        Q = 0.2 * np.eye(1)  # state weight matrix
        L = 0.1 * np.eye(self.dim_input)  # input weight matrix
        # add recursive constraints and costs
        for k in range(self.horizon):
            self.constraints += [ x[:,k+1] == self.vehicle.A*x[:,k]+self.vehicle.B*u[:,k] ]
            self.constraints += [ u[:,k] <= self.vehicle.max_input ]
            self.constraints += [ u[:,k] >= -self.vehicle.max_input ]
            # obstacles
            '''
            obstacles code
            '''
            # sum of cost
            self.cost += Q @ self.error_function(x[:,k])**2 + L @ (u[:,k])**2
            # terminal constraint
            '''
            terminal constaint
            '''
        constraints += [ x[:,0] == self.initial_state ]
    
        # Solves the problem
        problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        problem.solve(solver=cp.OSQP)