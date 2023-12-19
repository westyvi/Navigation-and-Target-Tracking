# -*- coding: utf-8 -*-
"""
Single Target Tracking Project:
    
    Guassian Mixture Probability Hypothetiy Density (GM-PHD) 
                Multi-target tracker

AEM667 - Navigation and Target Tracking
Project 5

Written by Joey Westermeyer 2023

notes:
    issues with S being non invertible. Investigate:
        Joseph form update (done, didn't help)
        iterate on H,K matrices - making EKF an IEKF - (improved singularity from t=14 to t=18)
        Add small number to diagonal of P and S to prevent underflow (didn't help)
        
        input scaling
        Different KF formulations (square root filter?)
        Thoronton and Bierman UD implementation
    would like to make plotting not limited to scatter
        write code that recognizes new tracks when plotting via distance change
    
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from p5_Classes import Gaussian, GMPHD
import os

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
x0 = np.array([0.0001, 0.0001, 0, 0, 0])  # initial state vector
weight0 = 10E-10
P0 = np.diag([.000001,.000001,.000001,.000001, 0.01])
initialPHD = [Gaussian(weight0, x0, P0)]
tracker = GMPHD(initialPHD)

# create 3D array to log state vectors of all targets through time
# third dimension is max number of targets for PHD cap step
# if targets don't exist, the states in this array will remain 0
xs = np.zeros((datalength, x0.shape[0], 100)) 
cardinalities = np.zeros(datalength)
xs[:] = np.nan
cardinalities[:] = np.nan

dt = 1/1  # measurements taken at 1 Hz # FIXME cheating

# iterate through time steps, feeding in recorded data
for i in range(datalength):
    print('')
    print(i)
 
    
    # define clean and cluttered sensor data
    bearings = dataframes['bearings'].iloc[i, :].dropna()
    ranges = dataframes['ranges'].iloc[i, :].dropna()
   
    # run tracker with sensor data for current time step
    output, N = tracker.run(dt, ranges/1000, bearings)
    print(len(output))

    # log tracker results
    cardinalities[i] = N
    if output is not None:
        j = 0
        for state in output:
            xs[i,:,j] = state
            xs[i,0:4,j] = xs[i,0:4,j]*1000
            j += 1
    

#%% plot 
filter_size = 8
truth_size = 5
opacity= .5

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
truth[:] = np.nan
true_cardinality = np.zeros(datalength)
true_cardinality[:] = np.nan
for i in range(datalength):
    states = dataframes['truth'].iloc[i,:].dropna()
    separated_states = [states[i:i + 5] for i in range(0, len(states), 5)]
    true_cardinality[i] = len(separated_states)
    for j in range(0,len(separated_states)):
        truth[i,:,j] = separated_states[j]
truth_time = np.linspace(0, datalength*dt, num=datalength)

# plot estimated xy plane track trajectory
fig, ax = plt.subplots()
# j is objects: loop through each potential object, plot it for all time (first index) and one state (middle index)
for j in range(0,100):
    if j == 0:
        ax.scatter(xs[:,0,j],xs[:,1,j], c='black', s=filter_size, label='cluttered PDAF', alpha=opacity)
        ax.scatter(truth[:,0,j],truth[:,1,j], c='red', s=truth_size, label='truth', alpha=opacity) 
    else:
        ax.scatter(xs[:,0,j],xs[:,1,j], c='black', s=filter_size, alpha=opacity)
        ax.scatter(truth[:,0,j],truth[:,1,j], c='red', s=truth_size, alpha=opacity) 
ax.set(xlabel = 'x, m', ylabel = 'y, m',
      title = 'xy plane track trajectory')
ax.legend()
plt.grid(True)


# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, figsize=(8, 5))
fig.suptitle("Position vs time")
for j in range(0,100):
    if j ==0:
        ax1.scatter(truth_time, xs[:,0,j], c='black', s=filter_size, label='cluttered PDAF', alpha=opacity)
        ax1.scatter(truth_time, truth[:,0,j], c='red', s=truth_size, label='truth', alpha=opacity)
    else:
        ax1.scatter(truth_time, xs[:,0,j], c='black', s=filter_size, alpha=opacity)
        ax1.scatter(truth_time, truth[:,0,j], c='red', s=truth_size, alpha=opacity)
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="x_pos (m)")
ax1.scatter(ts, x_coordinates, s=2, alpha=.2)

for j in range(0,100):
    ax2.scatter(truth_time, xs[:,1,j], c='black', s=filter_size, alpha=opacity)
    ax2.scatter(truth_time, truth[:,1,j], c='red', s=truth_size, alpha=opacity)
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="y_pos (m)")
fig.legend()
ax2.scatter(ts, y_coordinates, s=2, alpha=.2)

# Plot the true and estimated cardinality vs. time
fig, ax1 = plt.subplots()
fig.suptitle("cardinality vs time")
ax1.plot(true_cardinality, 'r', label='truth')
ax1.plot(cardinalities, 'b', label='GM-PHD')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="Cardinality")
fig.legend()
ax1.set_ylim(bottom=0, top=None)

'''
# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle("Velocity vs time")
for j in range(0,100):
    #ax1.plot(xs[:,2,j], 'c', label='cluttered PDAF')
    ax1.plot(truth[:,2,j], c='red', s=5)
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="v_x (m/s)")

for j in range(0,100):
    #ax2.plot(xs[:,3,j], 'c', label='cluttered PDAF')
    ax2.plot(truth[:,3,j], c='red', s=5)
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="v_y (m/s)")
fig.legend()'''

# Plot the true and estimated turn-rate, 𝜔, vs. time
fig, ax1 = plt.subplots()
fig.suptitle("Turn rate vs time")
for j in range(0,100):
    if j ==0:
        ax1.scatter(truth_time, xs[:,4,j], c='black', s=filter_size, label='cluttered PDAF', alpha=opacity)
        ax1.scatter(truth_time, truth[:,4,j], c='red', s=truth_size, label='truth', alpha=opacity)
    else:
        ax1.scatter(truth_time, xs[:,4,j], c='black', s=filter_size, alpha=opacity)
        ax1.scatter(truth_time, truth[:,4,j], c='red', s=truth_size, alpha=opacity)
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="w (rad/s)")
fig.legend()

dummyvar = 5 # stops code from printing commented block of code to console
