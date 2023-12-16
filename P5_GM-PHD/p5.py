# -*- coding: utf-8 -*-
"""
Single Target Tracking Project:
    
    Guassian Mixture Probability Hypothetiy Density (GM-PHD) 
                Multi-target tracker

AEM667 - Navigation and Target Tracking
Project 5

Written by Joey Westermeyer 2023
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from p5_Classes import Gaussian, GMPHD
import os
import copy

# Load csv files into dictionary of data frames
resource_path = os.path.join(".", "project5_resources")
filenames = [f for f in os.listdir(resource_path) if f.endswith(".csv")]
dataframes = {}
for filename in filenames:
    basename = os.path.splitext(filename)[0]
    dataframes[basename] = pd.read_csv(
        os.path.join(resource_path, filename), header=None)
datalength = dataframes['truth'].shape[0]

# intialize PHD with one target with low weight
x0 = np.array([0.1, 0.1, 0, 0, 0])  # initial state vector
weight0 = 10E-10
P0 = np.diag([1,1,1,1, 0.01])
initialPHD = [Gaussian(weight0, x0, P0)]
tracker = GMPHD(initialPHD)

# create 3D array to log state vectors of all targets through time
# third dimension is max number of targets for PHD cap step
# if targets don't exist, the states in this array will remain 0
xs = np.zeros((datalength, x0.shape[0], 100)) 

dt = 1/1  # measurements taken at 1 Hz


for i in range(datalength):
    print(i)
    
    # define clean and cluttered sensor data
    bearings = dataframes['bearings'].iloc[i, :].dropna()
    ranges = dataframes['ranges'].iloc[i, :].dropna()
   
    output = tracker.run(dt, ranges, bearings)
    if output is not None:
        for j, state in output:
            xs[i,:,j] = state
            
    print(tracker.run(dt, ranges, bearings))
    

#%% plot 

### convert cluttered sensor data to xy for plotting
xy_coordinates = []

# Iterate through each time step
for i in range(datalength):
    # Extract range and bearing measurements for the current time step
    bearing_measurements = dataframes['bearings'].iloc[i].values[1:]
    range_measurements = dataframes['ranges'].iloc[i].values[1:]
    
    # Remove NaN values from range and bearing measurements
    mask = ~np.isnan(range_measurements)
    range_measurements = range_measurements[mask]
    bearing_measurements = bearing_measurements[mask]

    # Convert range and bearing measurements to x,y coordinates
    j = 0
    for r in range_measurements:
        bear = bearing_measurements[j]
        x = math.sqrt(r**2/(((math.tan(bear))**-2)+1))*math.tan(bear)/abs(math.tan(bear))
        y = math.sqrt(r**2/(((math.tan(bear))**2)+1))
        xy_coordinates.append([x, y, i])
        j += 1

# Convert the list of x,y,timestamp coordinates to a NumPy array
xy_coordinates = np.array(xy_coordinates)
x_coordinates = xy_coordinates[:, 0]
y_coordinates = xy_coordinates[:, 1]
ts = xy_coordinates[:,2]

### convert truth data into something plottable
truth = np.zeros((datalength, x0.shape[0], 100)) 
for i in range(datalength):
    states = dataframes['truth'].iloc[i,:].dropna()
    separated_states = [states[i:i + 5] for i in range(0, len(states), 5)]
    for j in range(0,len(separated_states)):
        truth[i,:,j] = separated_states[j]

# plot estimated xy plane track trajectory
fig, ax = plt.subplots()
for j in range(0,100):
    ax.plot(xs[:,0,j],xs[:,1,j], 'c', label='cluttered PDAF')
    ax.plot(truth[:,0,j],truth[:,1,j], 'r', label='truth')
ax.set(xlabel = 'x, m', ylabel = 'y, m',
      title = 'xy plane track trajectory')
ax.legend()
plt.grid(True)


# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, figsize=(8, 5))
fig.suptitle("Position vs time")
for j in range(0,100):
    ax.plot(xs[:,0,j], 'c', label='cluttered PDAF')
    ax.plot(truth[:,0,j], 'r', label='truth')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="x_pos (m)")
ax1.scatter(ts, x_coordinates, s=2, alpha=.2)

for j in range(0,100):
    ax.plot(xs[:,1,j], 'c', label='cluttered PDAF')
    ax.plot(truth[:,1,j], 'r', label='truth')
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="y_pos (m)")
fig.legend()
ax2.scatter(ts, y_coordinates, s=2, alpha=.2)

# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle("Velocity vs time")
for j in range(0,100):
    ax.plot(xs[:,2,j], 'c', label='cluttered PDAF')
    ax.plot(truth[:,2,j], 'r', label='truth')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="v_x (m/s)")

for j in range(0,100):
    ax.plot(xs[:,3,j], 'c', label='cluttered PDAF')
    ax.plot(truth[:,3,j], 'r', label='truth')
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="v_y (m/s)")
fig.legend()

# Plot the true and estimated turn-rate, 𝜔, vs. time
fig, ax1 = plt.subplots()
fig.suptitle("Turn rate vs time")
for j in range(0,100):
    ax.plot(xs[:,4,j], 'c', label='cluttered PDAF')
    ax.plot(xs[:,4,j], 'r', label='truth')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="w (rad/s)")
fig.legend()


# Plot the true and estimated cardinality vs. time
fig, ax1 = plt.subplots()
fig.suptitle("cardinality vs time")
for j in range(0,100):
    ax.plot(xs[:,4,j], 'c', label='cluttered PDAF')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="w (rad/s)")
fig.legend()
