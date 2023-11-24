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
from p4_Classes import EKF, NN
import os
import copy

a = np.array([1,2,3,4,5])
b = np.array([2,4,6,8,10])
fig, ax = plt.subplots()
ax.plot(a,b)

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
dt = 1/1  # measurements taken at 1 Hz

for i in range(datalength):

    bearings = dataframes['bearings_clean'].iloc[i, :].dropna()
    ranges = dataframes['ranges_clean'].iloc[i, :].dropna()
    xs_cleanEKF[i,:] = NN.runNNEKF(ekf_clean, dt, ranges, bearings)
    '''
    clutter_bearings = dataframes['bearings_clutter'].iloc[i, :].dropna()
    clutter_ranges = dataframes['ranges_clutter'].iloc[i, :].dropna()
    xs_clutterEKF[i,:] = NN.runNNEKF(ekf_clutter, dt, ranges, bearings)
    '''
    
    # call PDAF


# plot results for cluttered sensor data with missed detections


# plot estimated xy plane track trajectory for positive y

# first, convert clean sensor data to state to plot as points to compare to EKF performance
measures = np.zeros((datalength, 2)) # x,y 
for i in range(datalength):
    bear = dataframes['bearings_clean'].iloc[i]
    r = dataframes['ranges_clean'].iloc[i]
    measures[i,0] = math.sqrt(r**2/(((math.tan(bear))**-2)+1))*math.tan(bear)/abs(math.tan(bear))
    measures[i,1] = math.sqrt(r**2/(((math.tan(bear))**2)+1))
    
fig, ax = plt.subplots()
ax.plot(dataframes['truth'].iloc[:, 0],
        dataframes['truth'].iloc[:, 2], 'r', label='truth')
#ax.scatter(measures[:,0], measures[:,1], marker='o',s=1, color='b')
#ax.plot(xs_cleanEKF[:,0], xs_cleanEKF[:,1], 'b', label='clean NN EKF')
#ax.plot(xs_clutterEKF[:,0], xs_clutterEKF[:,1], 'g', label='cluttered NN EKF')
#ax.set(xlabel = 'x, m', ylabel = 'y, m',
 #      title = 'xy plane track trajectory')
#ax.legend()
plt.grid(True)

# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
# Plot the true and estimated turn-rate, 𝜔, vs. time