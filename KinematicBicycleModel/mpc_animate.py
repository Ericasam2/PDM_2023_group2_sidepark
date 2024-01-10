# pylint: skip-file
from csv import reader
from dataclasses import dataclass
from math import radians

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from kinetic_bicycle_model import KinematicBicycleModel
from libs import CarDescription, StanleyController, generate_cubic_spline

from MPC_controller import MPC_controller

import numpy as np


class Simulation:

    def __init__(self):
        '''
        the properties of the simulation window
        '''
        fps = 50.0

        self.dt = 1/fps
        self.map_size_x = 70
        self.map_size_y = 40
        self.frames = 1000  # simulation horizon
        self.loop = False


class Path:

    def __init__(self):

        # Get path to waypoints.csv
        data_path = 'data/sine_wave_waypoints.csv'
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
        self.initial_state = np.array([init_x, init_y, init_yaw, 0.0])
        self.current_state = self.initial_state
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

    def get_mpc(self, model, mpc, estimator, simulator):
        self.mpc = mpc
        self.estimator = estimator
        self.simulator = simulator
        
    def CL_drive(self):
        inputs = self.mpc.make_step(self.current_state)
        y_next = self.simulator.make_step(inputs)
        self.current_state = self.estimator.make_step(y_next)
        
        self.velocity, self.x, self.y, self.yaw = self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]
        self.steering_angle = inputs[1]
        
    def drive(self):
        
        acceleration = 0 if self.time > self.time_to_reach_target_velocity else self.get_required_acceleration()
        self.wheel_angle, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.velocity, self.wheel_angle)
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(self.x, self.y, self.yaw, self.velocity, acceleration, self.wheel_angle)

        print(f"Cross-track term: {self.crosstrack_error}{' '*10}", end="\r")
        

class StaticObstacle:
    
    def __init__(self):
        self.number = 0
        self.shape = []
        self.description = []
        self.obstacles = []

    class rectangle:
        '''
        rectangle shape static obstacle
        '''
        def __init__(self, v, h, w, th=0.0):
            self.vertex = v
            self.height = h
            self.width = w
            self.angle = th
            
        def get_obstacle(self):
            return [self.vertex, self.height, self.width, self.angle]

        def plot(self):
            return plt.Rectangle(self.vertex, self.width, self.height, color='blue', fill=True)
    
    class circle:
        '''
        circle shape static obstacle
        '''
        def __init__(self, c, r):
            self.center = c
            self.radius = r
            
        def get_obstacle(self):
            return [self.center, self.radius]
        
        def plot(self):
            return plt.Circle(self.center, self.radius, color='green', fill=True)

    def get_obstacle(self):
        # Get static obstacle data
        data_path = 'data/static_obstacles.csv'
        with open(data_path, newline='') as f:
            rows = list(reader(f, delimiter=','))[1:-1]
        self.number = len(rows)
        for s,d in rows:
            self.shape.append(s)
            self.description.append(eval(d))
        
        for i in range(self.number):
            if (self.shape[i] == 'rectangle'):
                v = self.description[i][0]
                h = self.description[i][1]
                w = self.description[i][2]
                self.obstacles.append(self.rectangle(v,h,w))
            else:
                c = self.description[i][0]
                r = self.description[i][1]
                self.obstacles.append(self.circle(c,r))
    
    def plot_obstacle(self, ax):
        for i in range(self.number):
            ax.add_patch(self.obstacles[i].plot())
        


@dataclass
class Fargs:
    ax: plt.Axes
    sim: Simulation
    path: Path
    car: Car
    car_outline: plt.Line2D
    front_right_wheel: plt.Line2D
    front_left_wheel: plt.Line2D
    rear_right_wheel: plt.Line2D
    rear_left_wheel: plt.Line2D
    rear_axle: plt.Line2D
    annotation: plt.Annotation
    target: plt.Line2D
   

def animate(frame, fargs):

    ax                = fargs.ax
    sim               = fargs.sim
    path              = fargs.path
    car               = fargs.car
    car_outline       = fargs.car_outline
    front_right_wheel = fargs.front_right_wheel
    front_left_wheel  = fargs.front_left_wheel
    rear_right_wheel  = fargs.rear_right_wheel
    rear_left_wheel   = fargs.rear_left_wheel
    rear_axle         = fargs.rear_axle
    annotation        = fargs.annotation
    target            = fargs.target

    # Camera tracks car
    ax.set_xlim(car.x - sim.map_size_x, car.x + sim.map_size_x)
    ax.set_ylim(car.y - sim.map_size_y, car.y + sim.map_size_y)

    # Drive and draw car
    car.CL_drive()
    outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.plot_car()
    car_outline.set_data(*outline_plot)
    front_right_wheel.set_data(*fr_plot)
    rear_right_wheel.set_data(*rr_plot)
    front_left_wheel.set_data(*fl_plot)
    rear_left_wheel.set_data(*rl_plot)
    rear_axle.set_data(car.x, car.y)

    # Show car's target
    target.set_data(path.px[car.target_id], path.py[car.target_id])

    # Annotate car's coordinate above car
    annotation.set_text(f'{float(car.x):.1f}, {float(car.y):.1f}, {float(car.yaw):.1f}')
    annotation.set_position((float(car.x), float(car.y) + 5))

    plt.title(f'{sim.dt*frame:.2f}s', loc='right')
    plt.xlabel(f'Speed: {float(car.velocity):.2f} m/s', loc='left')
    # plt.savefig(f'image/visualisation_{frame:03}.png', dpi=300)

    return car_outline, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel, rear_axle, target,


def main():
    
    sim  = Simulation()
    path = Path()
    car  = Car(0, 0, 0, path.px, path.py, path.pyaw, sim.dt)
    initial_state = np.array([0, 0, 0, 0])
    target_state = np.array([50, 10, 2, 0])
    controller = MPC_controller(car.kinematic_bicycle_model, initial_state, target_state)
    [model, mpc, estimator, simulator] = controller.generate()
    
    # Initial state
    e = np.ones([model.n_x,1])
    x0 = initial_state
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0
    
    # Use initial state to set the initial guess.
    mpc.set_initial_guess()
    
    car.get_mpc(model, mpc, estimator, simulator)
    
    obstacle = StaticObstacle()
    obstacle.get_obstacle()

    interval = sim.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    # Draw the path and road
    # road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)  # road
    # ax.add_patch(road)
    ax.plot(path.px, path.py, '--', color='gold')  # path
    
    # Draw the obstacles
    obstacle.plot_obstacle(ax)

    empty              = ([], [])
    target,            = ax.plot(*empty, '+r')
    car_outline,       = ax.plot(*empty, color=car.colour)
    front_right_wheel, = ax.plot(*empty, color=car.colour)
    rear_right_wheel,  = ax.plot(*empty, color=car.colour)
    front_left_wheel,  = ax.plot(*empty, color=car.colour)
    rear_left_wheel,   = ax.plot(*empty, color=car.colour)
    rear_axle,         = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)
    annotation         = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)

    fargs = [Fargs(
        ax=ax,
        sim=sim,
        path=path,
        car=car,
        car_outline=car_outline,
        front_right_wheel=front_right_wheel,
        front_left_wheel=front_left_wheel,
        rear_right_wheel=rear_right_wheel,
        rear_left_wheel=rear_left_wheel,
        rear_axle=rear_axle,
        annotation=annotation,
        target=target
    )]

    anim = FuncAnimation(fig, animate, frames=sim.frames, init_func=lambda: None, fargs=fargs, interval=interval, repeat=sim.loop)
    
    plt.grid()
    plt.show()
    # anim.save('animation.gif', writer='imagemagick', fps=50)




if __name__ == '__main__':
    main()
