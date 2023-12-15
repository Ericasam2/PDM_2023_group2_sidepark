import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KDTree

class RRTStar:
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01, rewire_count=None):
        self.X = X  # Search Space
        self.Q = Q  # List of lengths of edges added to tree
        self.x_init = np.array(x_init)  # Initial location
        self.x_goal = np.array(x_goal)  # Goal location
        self.max_samples = max_samples  # Max number of samples to take
        self.r = r  # Resolution of points to sample along edge when checking for collisions
        self.prc = prc  # Probability of checking whether there is a solution
        self.rewire_count = rewire_count if rewire_count is not None else 0
        self.c_best = float('inf')  # Length of best solution thus far
        self.trees = {0: {"V": [self.x_init], "E": {}}}

    def add_vertex(self, tree, vertex):
        self.trees[tree]["V"].append(vertex)

    def add_edge(self, tree, parent, child):
        if child is not None:
            self.trees[tree]["E"][child] = parent

    def nearby(self, tree, x_new, count):
        all_vertices = np.array(self.trees[tree]["V"])
        distances = np.linalg.norm(all_vertices - x_new, axis=1)
        indices = np.argsort(distances)
        return all_vertices[indices[:count]]

    def new_and_near(self, tree, q):
        x_rand = self.sample_random_point()
        x_near = self.nearest_vertex(tree, x_rand)
        x_new = self.steer(x_near, x_rand, q)
        return x_new, x_near

    def sample_random_point(self):
        if np.random.rand() < self.prc:
            return np.copy(self.x_goal)
        else:
            return np.random.rand(2) * np.array([self.X.map_size_x, self.X.map_size_y])

    def nearest_vertex(self, tree, x_rand):
        all_vertices = np.array(self.trees[tree]["V"])
        distances = np.linalg.norm(all_vertices - x_rand, axis=1)
        min_index = np.argmin(distances)
        return all_vertices[min_index]

    def steer(self, x_near, x_rand, q):
        distance = euclidean(x_near, x_rand)
        if distance < q:
            return x_rand
        else:
            theta = np.arctan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])
            return x_near + q * np.array([np.cos(theta), np.sin(theta)])

    def connect_to_point(self, tree, x_near, x_new):
        step_size = 0.1
        direction = x_new - x_near
        distance = np.linalg.norm(direction)
        steps = int(distance / step_size)
        for i in range(1, steps + 1):
            intermediate_point = x_near + i * (step_size / distance) * direction
            if not self.X.collision_free(intermediate_point, x_new, self.r):
                return False
        return True

    def check_solution(self):
        for tree in self.trees:
            for vertex in self.trees[tree]["V"]:
                if euclidean(vertex, self.x_goal) < self.r:
                    return True, self.reconstruct_path(tree, vertex)
        return False, []

    def reconstruct_path(self, tree, goal_vertex):
        path = [goal_vertex]
        current_vertex = goal_vertex
        while current_vertex != self.x_init:
            current_vertex = self.trees[tree]["E"][current_vertex]
            path.append(current_vertex)
        return path[::-1]

    def get_nearby_vertices(self, tree, x_init, x_new):
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        L_near = [(self.path_cost(x_init, x_near) + self.segment_cost(x_near, x_new), x_near) for
                  x_near in X_near]
        L_near.sort(key=lambda x: x[0])
        return L_near

    def rewire(self, tree, x_new, L_near):
        for c_near, x_near in L_near:
            curr_cost = self.path_cost(self.x_init, x_near)
            tent_cost = self.path_cost(self.x_init, x_new) + self.segment_cost(x_new, x_near)
            if tent_cost < curr_cost and self.X.collision_free(x_near, x_new, self.r):
                self.trees[tree]["E"][x_near] = x_new

    def connect_shortest_valid(self, tree, x_new, L_near):
        for c_near, x_near in L_near:
            if c_near + self.cost_to_go(x_near, self.x_goal) < self.c_best and self.connect_to_point(tree, x_near, x_new):
                break

    def current_rewire_count(self, tree):
        if self.rewire_count is None:
            return len(self.trees[tree]["V"])
        return min(len(self.trees[tree]["V"]), self.rewire_count)

    def path_cost(self, x_init, x_goal):
        return euclidean(x_init, x_goal)

    def segment_cost(self, x_near, x_new):
        return euclidean(x_near, x_new)

    def cost_to_go(self, x, x_goal):
        return euclidean(x, x_goal)

    def rrt_star(self):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        for _ in range(self.max_samples):
            for q in self.Q:
                x_new, x_nearest = self.new_and_near(0, q)
                if x_new is None:
                    continue

                L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                self.connect_shortest_valid(0, x_new, L_near)

                if x_new in self.trees[0]["E"]:
                    self.rewire(0, x_new, L_near)

                solution = self.check_solution()
                if solution[0]:
                    return solution[1]

        return []

# Usage example:
# X is the search space with map_size_x and map_size_y attributes
# Q is the list of edge lengths
# collision_free should be a function that checks collision between two points
# You need to adapt this to fit your existing code
# rrt = RRTStar(X, Q, x_init=(0, 0), x_goal=(10, 10), max_samples=5000, r=1.0, prc=0.01)
# path = rrt.rrt_star()
