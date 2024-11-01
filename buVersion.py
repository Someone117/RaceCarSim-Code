"""
Autonomous RACECAR

File Name: demo2.py

Title: RACECAR program based on BU's F1Tenth code, with some optimizations to get it running in semi-real time

"""


import sys

sys.path.insert(0, "../library")
import racecar_core
import numpy as np
from astarAlg import astar
import math
import time

import matplotlib.pyplot as plt

rc = racecar_core.create_racecar()

position = [0, 0]
angle = 0
velocity = [0, 0]
im = None
ax = None

def lidar_to_local(
    robot_x: float, robot_y: float, robot_theta: float, lidar_samples: np.ndarray
) -> np.ndarray:
    num_samples = 720
    lidar_angles = np.deg2rad(
        np.linspace(180, -180, num_samples)
    )  # LIDAR angles from -180 to 180 degrees

    # Convert each LIDAR sample from polar (distance, angle) to Cartesian coordinates (x_local, y_local)
    x_local = -lidar_samples * np.cos(lidar_angles)
    y_local = lidar_samples * np.sin(lidar_angles)
    # Filter valid indices within bounds
    valid_mask = lidar_samples > 5

    # Apply mask to get valid points
    x_local = x_local[valid_mask]
    y_local = y_local[valid_mask]

    # Apply the robot's rotation and position to convert to global coordinates
    # global_x = robot_x + x_local * np.cos(robot_theta) - y_local * np.sin(robot_theta)
    # global_y = robot_y + x_local * np.sin(robot_theta) + y_local * np.cos(robot_theta)

    global_x = x_local + robot_y*58
    global_y = y_local

    return np.stack((global_x, global_y), axis=-1)
    # return np.stack((x_local, y_local), axis=-1)


# GaussianMap class definition
class GaussianMap:
    def __init__(self, x_res=800, y_res=800, sigma=10, decay_rate=0.99):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.gaussian_map = np.zeros((x_res, y_res))
        self.x_center = 100
        self.y_center = y_res // 2

    def update_gaussian_map(self, grid):
        # this is the major time usage
        self.gaussian_map *= self.decay_rate

        x_grid, y_grid = np.meshgrid(np.arange(self.x_res), np.arange(self.y_res))
        gaussian_kernel = np.exp(
            -((x_grid - self.x_res // 2) ** 2 + (y_grid - self.y_res // 2) ** 2)
            / (2 * self.sigma**2)
        )


        indices = np.argwhere(grid == 1)

        for i, j in indices: 
            # Shift the Gaussian kernel to the current point
            kernel_shifted = np.roll(
                np.roll(gaussian_kernel, j - self.x_res // 2, axis=1),
                i - self.y_res // 2,
                axis=0,
            )
            self.gaussian_map += kernel_shifted

    def visualize_gaussian_map(self):
        im.set_array(self.gaussian_map)
        plt.draw()
        plt.pause(0.01)


grid_size = 800
occupancy_grid = np.zeros((800, 800), int)


def update_lidar_and_visualize():
    global position
    global velocity
    global angle
    global occupancy_grid
    global grid_size

    # # get global position, angle and LIDAR data
    # angle = angle + rc.physics.get_angular_velocity()[1]*rc.get_delta_time()

    samples = lidar_to_local(position[0], position[1], angle, rc.lidar.get_samples())

    # rotation_matrix = np.array(
    #     [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    # )
    acc_local = [
        rc.physics.get_linear_acceleration()[0],
        rc.physics.get_linear_acceleration()[2],
    ]
    # acc_global = np.dot(rotation_matrix, acc_local)
    acc_global = acc_local
    velocity[0] = velocity[0] + acc_global[0] * rc.get_delta_time()
    velocity[1] = velocity[1] + acc_global[1] * rc.get_delta_time()
    position[0] = position[0] + velocity[0] * rc.get_delta_time()
    position[1] = position[1] + velocity[1] * rc.get_delta_time()

    # plt.scatter(samples[:, 1],samples[:, 0], c='r', label="Walls")
    # plt.scatter(*position, c='g', label="Start")
    # # plt.scatter(*goal_pos, c='b', label="Goal")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Convert samples to array for efficient processing
    samples = np.array(samples)

    # Add offset and convert to integer indices
    x_coords = (samples[:, 0] + 200).astype(int)
    y_coords = (samples[:, 1] + 400).astype(int)

    # Filter valid indices within bounds
    valid_mask = (
        (x_coords >= 0)
        & (x_coords < grid_size)
        & (y_coords >= 0)
        & (y_coords < grid_size)
    )

    # Apply mask to get valid points
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Update occupancy grid using valid points
    occupancy_grid[x_coords, y_coords] = 1

    # im.set_array(occupancy_grid)
    # plt.draw()
    # plt.pause(0.01)
    gaussian_map.update_gaussian_map(occupancy_grid)
    gaussian_map.visualize_gaussian_map()


def start():
    global position
    global velocity
    global angle
    global im
    global ax
    plt.ion()
    plt.show()
    fig, ax = plt.subplots()
    buffered = np.random.rand(10, 10)
    plt.title("Gaussian Map with Decay")
    im = ax.imshow(buffered)


def update():
    update_lidar_and_visualize()
    rc.drive.set_speed_angle(0.25, 0)

# Initialize Gaussian map
gaussian_map = GaussianMap(sigma=8, decay_rate=0.98)

# Register start and update functions
rc.set_start_update(start, update)
rc.go()
