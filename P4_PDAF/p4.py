# -*- coding: utf-8 -*-
"""
Single Target Tracking Project:
    
    Probability Density Association Filter (PDAF) 
                        vs
            EKF with nearest neighbor

AEM667 - Navigation and Target Tracking
Project 4

Written by Joey Westermeyer 2023
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from p4_Classes import EKF, NN, PDAF
import os
import copy
import matplotlib
matplotlib.use('Agg')


# Load csv files into dictionary of data frames
resource_path = os.path.join(".", "project4_resources")
filenames = [f for f in os.listdir(resource_path) if f.endswith(".csv")]
dataframes = {}
for filename in filenames:
    basename = os.path.splitext(filename)[0]
    dataframes[basename] = pd.read_csv(
        os.path.join(resource_path, filename), header=None)
datalength = dataframes['truth'].shape[0]

# intialize target state
xpos = 500  # m
xdot = 7.5  # m/s
ypos = 500  # m
ydot = 7.5  # m/s
w = math.radians(2)  # rad/s
x0 = np.array([xpos, ypos, xdot, ydot, w])  # initial state vector
xs_cleanEKF = np.zeros((datalength, x0.shape[0])) # empty 2D array to log all state vectors through time
xs_clutterEKF = np.zeros((datalength, x0.shape[0]))
xs_cleanPDAF  = np.zeros((datalength, x0.shape[0]))
xs_clutterPDAF = np.zeros((datalength, x0.shape[0]))

# initialize EKF
sigma_x = 10  # x uncertainty, m
sigma_y = 10  # y uncertainty, m
sigma_xdot = 5  # xdot uncertainty, m/s
sigma_ydot = 5  # ydot uncertainty, m/s
sigma_w = math.radians(2) # omega uncertainty, rad/s
P0 = np.diag([sigma_x**2, sigma_y**2, sigma_xdot **
             2, sigma_ydot**2, sigma_w**2])
ekf_clean = EKF(x0, P0)
ekf_clutter = EKF(copy.deepcopy(x0), copy.deepcopy(P0))
pdaf_clean = PDAF(copy.deepcopy(x0), copy.deepcopy(P0))
pdaf_clutter = PDAF(copy.deepcopy(x0), copy.deepcopy(P0))
dt = 1/1  # measurements taken at 1 Hz

for i in range(datalength):
    
    # define clean and cluttered sensor data
    bearings_clean = dataframes['bearings_clean'].iloc[i, :].dropna()
    ranges_clean = dataframes['ranges_clean'].iloc[i, :].dropna()
    bearings_clutter = dataframes['bearings_clutter'].iloc[i, :].dropna()
    ranges_clutter = dataframes['ranges_clutter'].iloc[i, :].dropna()
    
    # run clean EKF
    #xs_cleanEKF[i,:] = NN.runNNEKF(ekf_clean, dt, ranges_clean, bearings_clean)
    
    # run cluttered EKF
    xs_clutterEKF[i,:] = NN.runNNEKF(ekf_clutter, dt, ranges_clutter, bearings_clutter)
    
    # run clean PDAF
    #xs_cleanPDAF[i,:] = pdaf_clean.runPDAF(dt, ranges_clean, bearings_clean)
   
    # run cluttered PDAF
    xs_clutterPDAF[i,:] = pdaf_clutter.runPDAF(dt, ranges_clutter, bearings_clutter)
    

# plot estimated xy plane track trajectory
# first, convert clean sensor data to state to plot as points to compare to EKF performance
measures = np.zeros((datalength, 2)) # x,y 
for i in range(datalength):
    bear = dataframes['bearings_clean'].iloc[i].iloc[0]
    r = dataframes['ranges_clean'].iloc[i].iloc[0]
    measures[i,0] = math.sqrt(r**2/(((math.tan(bear))**-2)+1))*math.tan(bear)/abs(math.tan(bear))
    measures[i,1] = math.sqrt(r**2/(((math.tan(bear))**2)+1))
    
# convert cluttered sensor data to xy for plotting
# Create an empty list to store all x,y coordinates
xy_coordinates = []

# Iterate through each time step
for i in range(datalength):
    # Extract range and bearing measurements for the current time step
    bearing_measurements = dataframes['bearings_clutter'].iloc[i].values[1:]
    range_measurements = dataframes['ranges_clutter'].iloc[i].values[1:]
    
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


#%% plot results for cluttered sensor data with missed detections:

# Plot 1: original xy_trajectory (filter and truth)
fig, ax = plt.subplots()
ax.plot(dataframes['truth'].iloc[:, 0],
      dataframes['truth'].iloc[:, 2], 'r', label='truth')
#ax.plot(xs_cleanEKF[:,0], xs_cleanEKF[:,1], 'b', label='clean NN EKF')
ax.plot(xs_clutterEKF[:,0], xs_clutterEKF[:,1], 'g', label='cluttered NN EKF')
#ax.plot(xs_cleanPDAF[:,0],xs_cleanPDAF[:,1], 'b', label='clean PDAF')
ax.plot(xs_clutterPDAF[:,0],xs_clutterPDAF[:,1], 'c', label='cluttered PDAF')
ax.set(xlabel = 'x, m', ylabel = 'y, m',
    title = 'xy plane track trajectory')
ax.legend()
plt.grid(True)
plt.savefig('plots/xy_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: xy_trajectory with all position measurements (clutter/noise)
fig, ax = plt.subplots()
ax.plot(dataframes['truth'].iloc[:, 0],
      dataframes['truth'].iloc[:, 2], 'r', label='truth')
ax.plot(xs_clutterEKF[:,0], xs_clutterEKF[:,1], 'g', label='cluttered NN EKF')
ax.plot(xs_clutterPDAF[:,0],xs_clutterPDAF[:,1], 'c', label='cluttered PDAF')
ax.scatter(x_coordinates, y_coordinates, s=2, alpha=0.2, c='blue', label='measurements')
ax.set(xlabel = 'x, m', ylabel = 'y, m',
    title = 'xy plane track trajectory with measurements')
ax.legend()
plt.grid(True)
plt.savefig('plots/xy_trajectory_with_measurements.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, figsize=(8, 5))
fig.suptitle("Position vs time")
ax1.plot(dataframes['truth'].iloc[:, 0], 'r', label='truth')
ax1.plot(xs_clutterEKF[:,0], 'g', label='cluttered NN EKF')
#ax1.plot(xs_cleanPDAF[:,0], 'b', label='clean PDAF')
ax1.plot(xs_clutterPDAF[:,0], 'c', label='cluttered PDAF')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="x_pos (m)")
ax1.scatter(ts, x_coordinates, s=2, alpha=.2)

ax2.plot(dataframes['truth'].iloc[:, 2], 'r')
ax2.plot(xs_clutterEKF[:,1], 'g')
#ax2.plot(xs_cleanPDAF[:,1], 'b')
ax2.plot(xs_clutterPDAF[:,1], 'c')
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="y_pos (m)")
fig.legend()
ax2.scatter(ts, y_coordinates, s=2, alpha=.2)
plt.savefig('plots/pos_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle("Velocity vs time")
ax1.plot(dataframes['truth'].iloc[:, 1], 'r', label='truth')
ax1.plot(xs_clutterEKF[:,2], 'g', label='cluttered NN EKF')
#ax1.plot(xs_cleanPDAF[:,2], 'b', label='clean PDAF')
ax1.plot(xs_clutterPDAF[:,2], 'c', label='cluttered PDAF')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="v_x (m/s)")

ax2.plot(dataframes['truth'].iloc[:, 3], 'r')
ax2.plot(xs_clutterEKF[:,3], 'g' )
#ax2.plot(xs_cleanPDAF[:,3], 'b')
ax2.plot(xs_clutterPDAF[:,3], 'c')
ax2.plot()
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="v_y (m/s)")
fig.legend()
plt.savefig('plots/vel_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the true and estimated turn-rate, 𝜔, vs. time
fig, ax1 = plt.subplots()
fig.suptitle("Turn rate vs time")
ax1.plot(dataframes['truth'].iloc[:, 4], 'r', label='truth')
ax1.plot(xs_clutterEKF[:,4], 'g', label='cluttered NN EKF')
#ax1.plot(xs_cleanPDAF[:,4], 'b', label='clean PDAF')
ax1.plot(xs_clutterPDAF[:,4], 'c', label='cluttered PDAF')
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="w (rad/s)")
fig.legend()
plt.savefig('plots/turnrate_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()