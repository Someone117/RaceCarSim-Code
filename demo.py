"""
Autonomous RACECAR
MIT License

File Name: demo2.py

Title: RACECAR program that runs on 60hz LIDAR, also dispalys debug info 
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(0, '../library')
import racecar_core
import numpy as np
from astar2 import astar2
import math
import time
from matplotlib import colors

import matplotlib.pyplot as plt
from scipy.spatial import KDTree

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
counter = 0
isDriving = False
position = [0,0]
angle = 0
velocity = [0,0]
im = None
ax = None

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    # If we use a global variable in our function, we must list it at
    # the beginning of our function like this
    global counter
    global isDriving
    global position
    global velocity
    global angle
    global im
    global ax

    # The start function is a great place to give initial values to global variables
    counter = 0
    isDriving = False

    # This tells the car to begin at a standstill
    rc.drive.set_max_speed(1.0)
    rc.drive.stop()
    plt.ion()
    plt.show()
    fig, ax = plt.subplots()
    # plt.imshow(buffered, cmap='viridis', origin='upper')
    buffered = np.random.rand(10, 10)
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]  # Define boundaries for each value
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(buffered, cmap=cmap, norm=norm)
    

def lidar_to_local(robot_x: float, robot_y: float, robot_theta: float, lidar_samples: np.ndarray) -> np.ndarray:
    num_samples = 720
    lidar_angles = np.deg2rad(np.linspace(180, -180, num_samples))  # LIDAR angles from -180 to 180 degrees
    
    # Convert each LIDAR sample from polar (distance, angle) to Cartesian coordinates (x_local, y_local)
    x_local = -lidar_samples * np.cos(lidar_angles)
    y_local = lidar_samples * np.sin(lidar_angles)

    # Filter valid indices within bounds
    valid_mask = (lidar_samples > 5)

    # Apply mask to get valid points
    x_local = x_local[valid_mask]
    y_local = y_local[valid_mask]

    # # Apply the robot's rotation and position to convert to global coordinates
    # global_x = robot_x + x_local * np.cos(robot_theta) - y_local * np.sin(robot_theta)
    # global_y = robot_y + x_local * np.sin(robot_theta) + y_local * np.cos(robot_theta)

    # return np.stack((global_x, global_y), axis=-1)
    return np.stack((x_local, y_local), axis=-1)

def find_sequences_numpy(array):
    array = np.array(array)  # Convert the input to a NumPy array
    rows, cols = array.shape

    # Initialize an array to hold the results: [start position, length] for each row
    result = []  # Default values of -1 if no sequence found
    
    for row_index in range(rows):
        row = array[row_index]
        
        # Find all the indices of 1's in the row
        ones_indices = np.where(row == 1)[0]
        
        if len(ones_indices) < 2:
            # If there are fewer than 2 ones, no valid sequence exists
            continue
        else:
            # Iterate through pairs of ones to find valid sequences
            maxstart = -1
            maxlength = -1
            for i in range(len(ones_indices) - 1):
                start = ones_indices[i]
                end = ones_indices[i + 1]

                if start+1 == end :
                    continue
                
                # Check if the segment between the two 1's contains only 0's
                if np.all(row[start + 1:end] == 0):
                    length = end - start + 1
                    if(length-2 > maxlength) :
                        maxstart = start
                        maxlength = length - 2
                    break
            if(maxstart > -1 and maxlength > -1) :
                if(maxlength > 5) :
                    result.append([maxstart + (maxlength /2), row_index, maxstart])

    
    return result

def add_circular_buffer(array, r):
    # Get the dimensions of the input array
    rows, cols = array.shape

    # Create a copy of the array to modify
    output = array.copy()

    # Iterate through each element of the array
    for i in range(rows):
        for j in range(cols):
            if array[i, j] == 1:
                # Add circular buffer of radius r around the current element
                for x in range(max(0, i - r), min(rows, i + r + 1)):
                    for y in range(max(0, j - r), min(cols, j + r + 1)):
                        if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                            if output[x, y] == 0:  # Set buffer to 2 only if it's currently 0
                                output[x, y] = 2
                # Keep the original element as 1
                output[i, j] = 1
    return output


def add_plus_buffer(array, r):
    # Create a copy of the array to modify
    output = array.copy()

    # Find the indices of elements equal to 1
    indices = np.argwhere(array == 1)

    for i, j in indices:
        # Create the '+' shaped buffer of radius r around the current element
        output[max(0, i - r):min(array.shape[0], i + r + 1), j] = 2
        output[i, max(0, j - r):min(array.shape[1], j + r + 1)] = 2
        # Keep the original element as 1
        output[i, j] = 1

    return output

def add_concentric_plus_buffers(array, r1, r2):
    # Create a copy of the array to modify
    output = array.copy()

    # Find the indices of elements equal to 1
    indices = np.argwhere(array == 1)

    for i, j in indices:
        # Add '+' shaped buffer of radius r1 with value 1
        output[max(0, i - r1):min(array.shape[0], i + r1 + 1), j] = 1
        output[i, max(0, j - r1):min(array.shape[1], j + r1 + 1)] = 1

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
    global position
    global angle
    global velocity
    global im
    global ax

    # get global position, angle and LIDAR data
    angle = angle + rc.physics.get_angular_velocity()[1]*rc.get_delta_time()

    samples = lidar_to_local(position[0], position[1], angle, rc.lidar.get_samples())

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    acc_local = [rc.physics.get_linear_acceleration()[0], rc.physics.get_linear_acceleration()[2]]
    acc_global = np.dot(rotation_matrix, acc_local)
    velocity[0] = velocity[0] + acc_global[0]*rc.get_delta_time()
    velocity[1] = velocity[1] + acc_global[1]*rc.get_delta_time()
    position[0] = position[0] + velocity[0]*rc.get_delta_time()
    position[1] = position[1] + velocity[1]*rc.get_delta_time()
    
    # plt.scatter(samples[:, 1],samples[:, 0], c='r', label="Walls")
    # plt.scatter(*position, c='g', label="Start")
    # # plt.scatter(*goal_pos, c='b', label="Goal")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    
    grid_size = 200
    occupancy_grid = np.zeros((grid_size, grid_size), int)

    # Convert samples to array for efficient processing
    samples = np.array(samples)

    # Add offset and convert to integer indices
    x_coords = (samples[:, 0]).astype(int) // 5
    y_coords = ((samples[:, 1] + 499).astype(int) // 5)

    # Filter valid indices within bounds
    valid_mask = (x_coords >= 0) & (x_coords < grid_size) & (y_coords >= 0) & (y_coords < grid_size)

    # Apply mask to get valid points
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Update occupancy grid using valid points
    occupancy_grid[x_coords, y_coords] = 1

    buffer_size = 1
    buffer_size_2 = 5
    start_t = time.time()
    # buffered = add_circular_buffer(occupancy_grid, buffer_size)
    # buffered = add_plus_buffer(occupancy_grid, buffer_size)
    buffered = add_concentric_plus_buffers(occupancy_grid, buffer_size, buffer_size_2)

    

    # print(time.time() - start_t)

    # for row in buffered :
    #     print(*row, sep=" ")

    start = (0,99)
    end = (199, 99)
    start_t = time.time()

    while (buffered[0,99] != 0 and buffered[1,99] != 0 and buffer_size_2 > buffer_size):
        buffer_size_2 = buffer_size_2 - 1
        buffered = add_concentric_plus_buffers(occupancy_grid, buffer_size, buffer_size_2)

    path = np.array(astar(buffered, start, end))
    # path = np.array(astar2(start, end, buffered))
    print(time.time() - start_t)

    # working above

    # print(len(path))
    for p in path :
        buffered[p[0]][p[1]] = 3


    im.set_array(buffered)
    plt.draw()
    plt.pause(0.01)


    car_size = 15
    if len(path) > car_size :
        heading = math.atan2(path[car_size][0], path[car_size][1] - 99)
    else :
        heading = 0

    angle = heading - np.pi/2

    # # reduce to a small amount of data: (line, midpoint) and also get the ones that are big
    # goals = np.array(find_sequences_numpy(occupancy_grid))

    # car_size = 20
    # max_distance = 5000

    # goals_correct = (goals * 5) - 499
    # goals_filtered = goals_correct[(goals_correct[:, 1] > (0))]
    
    
    # distances = np.sqrt(np.square(goals_filtered[:, 0]) + np.square(goals_filtered[:, 1]))
    # angles = (np.arctan2(goals_filtered[:, 1], goals_filtered[:, 0]) - (np.pi/2)) / np.pi
    
    # valid_mask = (distances > car_size) & (distances < max_distance)
    # distances_filtered = distances[valid_mask]
    # angles_filtered = angles[valid_mask]

    # # Combine into (distance, angle) pairs
    # distance_angle_pairs = np.column_stack((distances_filtered, angles_filtered))

    # Small epsilon to avoid division by zero for very close distances
    # epsilon = 1e-6

    # # Calculate weights (closer points have higher weights)
    # weights = 1 / (distances_filtered + epsilon)

    # # Normalize the weights so they sum to 1
    # weights_normalized = weights / np.sum(weights)

    # # Combine angles using a weighted average
    # combined_angle = np.sum(angles_filtered * weights_normalized)

    drive = 0.3
    # drive = 0
    multiplier = -4
    # print(angle)
    angle = max(min(angle*multiplier, 1), -1) # clamp to -1, 1 and invert angle and amplify angle
    # speed = 0.1 * (2.1-(abs(angle)*2))
    rc.drive.set_speed_angle(drive, angle)
    # print(walls)

    # plt.scatter(samples[:, 1],samples[:, 0], c='r', label="Walls")
    # plt.scatter(0, 0, c='g', label="Start")
    # plt.scatter(goals_filtered[:, 0],goals_filtered[:, 1], c='b', label="Goal")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # exit()


    # robot_radius = 0.2  # Define a buffer around the walls based on robot size

# Visualize the LIDAR points
    # for point in samples:
    #     x, y = int(point[0]*10)+50, int(point[1]*10)+50
    #     if 0 <= x < grid_size and 0 <= y < grid_size:
    #         occupancy_grid[x, y] = 1  # Mark occupied cells
    #         print('A')

    # # Reset the counter and start driving in an L every time the B button is pressed on
    # # the controller
    # if rc.controller.was_pressed(rc.controller.Button.B):
        
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
