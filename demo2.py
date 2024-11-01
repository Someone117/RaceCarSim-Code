"""
Autonomous RACECAR

File Name: demo2.py

Title: RACECAR program that runs on 10hz LIDAR

This version is optimized

"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(0, "../library")
import racecar_core
import numpy as np
from astarAlg import astar
import math
import time
from time import sleep
from matplotlib import colors

import matplotlib.pyplot as plt
from scipy.spatial import KDTree

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
counter = 5
isDriving = False
im = None
ax = None
path = []

########################################################################################
# Functions
########################################################################################


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    # If we use a global variable in our function, we must list it at
    # the beginning of our function like this
    global isDriving
    global im
    global ax

    # The start function is a great place to give initial values to global variables
    isDriving = False

    # This tells the car to begin at a standstill
    rc.drive.set_max_speed(1.0)  # unlocks car speed lol, this is probably bad
    rc.drive.stop()

    # display stuff
    plt.ion()
    plt.show()
    fig, ax = plt.subplots()
    buffered = np.random.rand(10, 10)
    cmap = colors.ListedColormap(["white", "red", "blue", "black"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(buffered, cmap=cmap, norm=norm)


def lidar_to_local(lidar_samples: np.ndarray) -> np.ndarray:
    # gets the lidar data
    num_samples = 720
    # rotate 90 degrees to the right
    # lidar_angles = np.deg2rad(np.linspace(270, -90, num_samples))
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

    return np.stack((x_local, y_local), axis=-1)


def add_concentric_plus_buffers(array, r1, r2):
    # create buffers like:
    #     |
    #   -----
    #     |
    # to optimize for writing to ram speed

    # Create a copy of the array to modify
    output = array.copy()

    # Find the indices of elements equal to 1
    indices = np.argwhere(array == 1)

    for i, j in indices:
        # Add '+' shaped buffer of radius r1 with value 1
        output[max(0, i - r1) : min(array.shape[0], i + r1 + 1), j] = 1
        output[i, max(0, j - r1) : min(array.shape[1], j + r1 + 1)] = 1

        # Keep the original element as 1
        output[i, j] = 1

        # Add '+' shaped buffer of radius r2 with value 2, excluding the r1 area
        for r in range(r1 + 1, r2 + 1):
            if i - r >= 0:
                output[i - r, j] = 2
            if i + r < array.shape[0]:
                output[i + r, j] = 2
            if j - r >= 0:
                output[i, j - r] = 2
            if j + r < array.shape[1]:
                output[i, j + r] = 2

    return output


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed
def update():
    global counter
    global isDriving
    global im
    global ax
    global path

    # every 1/10 of a sec, get LIDAR data
    if counter < 5:
        counter = counter + 1
    else:
        samples = lidar_to_local(rc.lidar.get_samples())
        counter = 0
        grid_size = 200
        # represents lidar data
        occupancy_grid = np.zeros((grid_size, grid_size), int)

        # Convert samples to array for efficient processing
        samples = np.array(samples)

        # Add offset and convert to integer indices
        x_coords = (samples[:, 0]).astype(int) // 5
        y_coords = (samples[:, 1] + 499).astype(int) // 5

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

        # create a buffer around the walls to not drive into them
        buffer_size = 1
        buffer_size_2 = 5
        buffer_size_final = buffer_size_2
        for buff in range(1, buffer_size_2 + buffer_size, 2):
            if occupancy_grid[0, 99 + buff] == 1 or occupancy_grid[0, 99 - buff] == 1:
                buffer_size_final = buff
                break

        buffered = add_concentric_plus_buffers(
            occupancy_grid, buffer_size, buffer_size_final
        )

        start = (0, 99)
        end = (199, 99)
        # A* to a point outside the LIDAR data to get a good path
        path = np.array(astar(buffered, start, end))

    # use data to go forward, don't care about changes in position due to driving and 10hz LIDAR, every 1/6th of a second the car cahnges by less than 5 cm
    car_size = 15

    if len(path) > car_size:
        heading = math.atan2(path[car_size][0], path[car_size][1] - 99)
    elif len(path) > 0:
        heading = math.atan2(path[len(path) - 1][0], path[len(path) - 1][1] - 99)
    else:
        heading = 0

    angle = heading - np.pi / 2  # set anlge to correct local

    drive = 0.25  # fast
    multiplier = -4
    angle = max(
        min(angle * multiplier, 1), -1
    )  # clamp to -1, 1 and invert angle and amplify angle
    rc.drive.set_speed_angle(drive, angle)


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # This prints a message every time that the right bumper is pressed during
    # a call to update_slow.  If we press and hold the right bumper, it
    # will print a message once per second
    if rc.controller.is_down(rc.controller.Button.RB):
        print("The right bumper is currently down (update_slow)")


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
