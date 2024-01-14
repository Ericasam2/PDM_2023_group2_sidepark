# pylint: skip-file
from csv import reader
from dataclasses import dataclass
from math import radians

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from kinematic_model import KinematicBicycleModel
from libs import CarDescription, StanleyController, generate_cubic_spline
from RRT_star.rrt_star import RRTStar
from RRT_star.search_space.search_space import SearchSpace
import numpy as np

class Simulation:

    def __init__(self):
        '''
        the properties of the simulation window
        '''
        fps = 50.0

        self.dt = 1/fps
        self.map_size_x = 25
        self.map_size_y = 25
        self.frames = 1000  # simulation horizon
        self.loop = False


class Path:

    def __init__(self, rrt):

        # RRTstar
        x = rrt.rrt_star()
        path = np.array(x)
        ds = 0.005

        self.px, self.py, self.pyaw, _ = generate_cubic_spline(path[:,0], path[:,1], ds)

    def path_append(self, path2, path3):

        goals = np.zeros([3,3])
        goals[0,:] = [self.px[-1], self.py[-1], self.pyaw[-1]]

        self.px = np.append(self.px, path2.px[1:])
        self.py = np.append(self.py, path2.py[1:])
        self.pyaw = np.append(self.pyaw, path2.pyaw[1:])
        
        goals[1,:] = [self.px[-1], self.py[-1], self.pyaw[-1]]

        self.px = np.append(self.px, path3.px[1:])
        self.py = np.append(self.py, path3.py[1:])
        self.pyaw = np.append(self.pyaw, path3.pyaw[1:])

        goals[2,:] = [self.px[-1], self.py[-1], self.pyaw[-1]]

        return goals
             

class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, delta_time, goals):

        # Model parameters
        self.x_init = init_x
        self.y_init = init_y
        self.x = init_x  # x coordinate
        self.y = init_y  # y coordinate
        self.yaw = init_yaw  # orientation
        self.delta_time = delta_time  # sampling time
        self.time = 0.0
        self.velocity = 0.0
        self.wheel_angle = 0.0
        self.angular_velocity = 0.0
        max_steer = radians(33)
        wheelbase = 2.96

        self.goals = goals
        self.goal_number = 0

        # Acceleration parameters
        self.target_velocity = 5.0
        self.time_to_reach_target_velocity = 2.0
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
        self.target_id = 0
        self.goal_reached = False

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
        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, max_steer, self.delta_time)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)


    def get_required_acceleration(self):
        remaining_distance = np.linalg.norm([self.x - self.goals[self.goal_number, 0], self.y - self.goals[self.goal_number, 1]])
        
        path_length = np.zeros((3))

        for i in range(self.goals.shape[0]):
            for j in range(self.px.shape[0]):
                if self.px[j] == self.goals[i, 0]:
                    path_length[i] = j
        
        halfway_point = path_length[self.goal_number] // 2
        if self.goal_number > 0:
            current_position = self.target_id - path_length[self.goal_number-1] 
        else: 
            current_position = self.target_id

        if remaining_distance <= 2:
            required_acceleration = 0
            self.velocity = 0
            
            self.goal_reached = True  # Set a flag indicating the goal is reached
            
        else:
            # Adjust acceleration based on position in the path
            if current_position <= halfway_point:
                # Accelerate towards halfway point
                required_acceleration = (self.target_velocity - self.velocity) / self.time_to_reach_target_velocity
            else:
                # Decelerate after halfway point
                required_acceleration = -self.velocity**2 / (2 * remaining_distance-2)
        
        if self.goal_number == 1:
            acceleration = - required_acceleration
        else:
            acceleration = required_acceleration

        return acceleration


    def plot_car(self):
        
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)


    def drive(self):
    
        if not self.goal_reached:
                acceleration = self.get_required_acceleration()
        else:
            self.goal_number += 1  # Move to next goal
            self.goal_reached = False  # Reset flag
            if self.goal_number == 2:
                acceleration = 0
            else:
                acceleration = self.get_required_acceleration()
            
        
        
        self.wheel_angle, self.target_id, self.crosstrack_error = self.tracker.stanley_control(
            self.x, self.y, self.yaw, self.velocity, self.wheel_angle
        )
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(
            self.x, self.y, self.yaw, self.velocity, acceleration, self.wheel_angle
        )

        print(f"Cross-track term: {self.crosstrack_error}{' ' * 10}", end="\r")



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
    ax = fargs.ax
    sim = fargs.sim
    car = fargs.car    
    car_outline = fargs.car_outline
    path = fargs.path
    front_right_wheel = fargs.front_right_wheel
    rear_right_wheel = fargs.rear_right_wheel
    front_left_wheel = fargs.front_left_wheel
    rear_left_wheel = fargs.rear_left_wheel
    rear_axle = fargs.rear_axle
    annotation = fargs.annotation
    target = fargs.target

    # Drive and draw car
    car.drive()
    outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = car.plot_car()
    car_outline.set_data(*outline_plot)
    front_right_wheel.set_data(*fr_plot)
    rear_right_wheel.set_data(*rr_plot)
    front_left_wheel.set_data(*fl_plot)
    rear_left_wheel.set_data(*rl_plot)
    rear_axle.set_data(car.x, car.y)

    plt.title(f'{sim.dt*frame:.2f}s', loc='right')
    plt.xlabel(f'Speed: {car.velocity:.2f} m/s', loc='left')

    # Show car's target
    target.set_data(path.px[car.target_id], path.py[car.target_id])

    # Annotate car's coordinate above car
    annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
    annotation.set_position((car.x, car.y + 5))
    

    # Check if the car has reached the goal
    goal_reached = np.linalg.norm([car.x - path.px[-1], car.y - path.py[-1]]) < 2

    if goal_reached:
        plt.title("Goal Reached!", loc='center')

        plt.grid()
        plt.pause(1)  # Pause for a moment before stopping the animation

        # Set repeat to False to stop the animation
        plt.close()
        sim.loop = False

    return car_outline, front_right_wheel, rear_right_wheel, front_left_wheel, rear_left_wheel, rear_axle, target,
 

def main():
    
    sim  = Simulation()
    X_dimesions = np.array([(-25, 25), (-25, 25)])
    obstacles = np.array([(-25,-25,-5,25), (5,-25,25,25), (1.75, 17.5, 4.25, 22.5)])
    searchspace = SearchSpace(X_dimesions, obstacles)
    start1 = (-2.5, -20)
    goal1 = (-0.75, 21.5)
    start2 = (-0.75, 21.5)
    goal2 = (2.5, 10)
    start3 = (2.5, 10)  # starting location
    goal3 = (2.5, 15)  # goal location

    Q = np.array([(1, 1)])  # length of tree edges
    r = 0.25  # length of smallest edge to check for intersection with obstacles
    max_samples = 2048  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    rrt = RRTStar(searchspace, Q, start1, goal1, max_samples, r, prc, rewire_count)
    rrt2 = RRTStar(searchspace, Q, start2, goal2, max_samples, r, prc, rewire_count)
    rrt3 = RRTStar(searchspace, Q, start3, goal3, max_samples, r, prc, rewire_count)


    path = Path(rrt)
    path2 = Path(rrt2)
    path3 = Path(rrt3)

    goals = path.path_append(path2, path3)

    car  = Car(start1[0], start1[1], np.pi/2, path.px, path.py, path.pyaw, sim.dt, goals)

    interval = sim.dt * 10**3


    # Create subplots
    plt.figure(num='RRT* and smoothing', figsize=(12, 5))  

    # Subplot 1: RRT* plots
    plt.subplot(1, 2, 1)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    rrt.plot_tree()
    rrt2.plot_tree()
    rrt3.plot_tree()
    plt.legend(labels=['RRT* vertices', 'edges'])

    # Plot obstacles in blue
    for obs in obstacles:
        plt.gca().add_patch(plt.Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], color='blue', alpha=0.3))

    plt.title('RRT* Plots')

    # Subplot 2: Path plots
    plt.subplot(1, 2, 2)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.plot(path.px, path.py, '--', color='red', label='path')  # path
    plt.plot(path2.px, path2.py, '--', color='red')  # path
    plt.plot(path3.px, path3.py, '--', color='red')  # path
    plt.legend()

    # Plot obstacles in blue
    for obs in obstacles:
        plt.gca().add_patch(plt.Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], color='blue', alpha=0.3))

    plt.title('Path Plots')

    # Add titles to the entire figure
    plt.suptitle('Combined RRT* and Path Plots')


    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the figure
    plt.show()

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    ax.plot(path.px, path.py, '--', color='gold')  # path
    ax.plot(path2.px, path2.py, '--', color='gold')  # path
    ax.plot(path3.px, path3.py, '--', color='gold')  # path
    
    # Draw the obstacles
    obstacle = StaticObstacle()
    obstacle.get_obstacle()
    obstacle.plot_obstacle(ax)

    empty              = ([], [])
    target,            = ax.plot(*empty, '+r')
    car_outline,       = ax.plot(*empty, color= 'red')
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

    _ = FuncAnimation(fig, animate, frames=sim.frames, init_func=lambda: None, fargs=fargs, interval=interval, repeat=sim.loop)
    
    plt.grid()
    plt.show()
    # anim.save('animation.gif', writer='imagemagick', fps=50)

    



if __name__ == '__main__':
    main()
    
