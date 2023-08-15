import torch
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def checkinfo():
    print("PyTorch Version:", torch.__version__) # Expected 2.0.1
    print("GPU Available:", torch.cuda.is_available()) # Expected 'cuda' or true
    print(torch.version.cuda)

#checkinfo()

# Config Torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def psuecode():
    # iteration 1:
    # get 3 points that create equilaterial triangle, centred on 0, 0

    # iteration 2:
    # get points array and for every point:
    # get the mid 1/3 and create a equilaterial triangle outwards :
    # union four different arrays together:
    # - the previous 
    # - the 1/3 point between two points from previous
    # - the 2/3 point between two points from previous
    # - the outward point between the 1/3 point and 2/3 that has a constant length each iteration
    # save new to previous

    # iteration 3:
    # iteration 2 steps again with previous points

    # plot using "plt.fill(x, y)"
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill.html
    pass

def inital_triangle():
    points = [(-0.5, -sqrt(3)/4), (0, sqrt(3)/4), (0.5, -sqrt(3)/4), (-0.5, -sqrt(3)/4)]
    return torch.Tensor(points).to(device)
    

def iteration(previous_points, drawing=False):
    #difference between 2 points / 3 then add to first
    # diff = next - previous
    difference = (previous_points[1:] - previous_points[:-1]) / 3
    first_points = previous_points[:-1] + difference
    second_points = (previous_points[:-1] + difference * 2)
    gap_size = sqrt(difference[0][0].item() ** 2 + difference[0][1].item() ** 2)
    
    # outward_point is 60 degrees left from first_point with distance difference
    # O/A
    angles = torch.atan2(difference[:, 1], difference[:, 0]) + (torch.pi / 3)
    outward_points = first_points + torch.stack((torch.cos(angles)*gap_size, torch.sin(angles)*gap_size), dim=1)    

    # Useful for debug to see the progression of 
    if (drawing):
        plt.scatter(previous_points[:, 0].cpu(), previous_points[:, 1].cpu(), color='b')
        plt.scatter(first_points[:, 0].cpu(), first_points[:, 1].cpu(), color='g')
        plt.scatter(outward_points[:, 0].cpu(), outward_points[:, 1].cpu(), color='r')
        plt.scatter(second_points[:, 0].cpu(), second_points[:, 1].cpu(), color='m')
        plt.show()

    test = torch.cat([
        previous_points[:-1, None, :], # previous
        first_points[:, None, :], # first points
        outward_points[:, None, :], # outward
        second_points[:, None, :] # second third points
        ], dim=2).view(-1, 2)
    return torch.cat([
        test, previous_points[-1:, :]])

def koch_snowflake(level=1):
    if level < 1:
        raise ValueError
    else:
        points = inital_triangle()
        current_level = 1
        print("Completed level:", current_level)
        while (level > current_level):
            if False:
                points = iteration(points, drawing=True)
            else:
                points = iteration(points)
            
            current_level = current_level + 1
            print("Completed level:", current_level)
        
        return points


points = koch_snowflake(4).cpu()
plt.fill(points[:, :1], points[:, 1:], "c")
plt.show()

# Possible improvements
# - Preallocate 4x bigger array instead of combine 2x speed on writes
# - Use gap size for calculating 1/3 and 2/3 points
# - Decrease floating points numbers (high ram usage/can't be seen in simulation anyway, currently float64)