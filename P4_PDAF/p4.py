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
xs = np.zeros((datalength, x0.shape[0])) # empty 2D array to log all state vectors through time

# initialize EKF
sigma_x = 10  # x uncertainty, m
sigma_y = 10  # y uncertainty, m
sigma_xdot = 5  # xdot uncertainty, m/s
sigma_ydot = 5  # ydot uncertainty, m/s
sigma_w = math.radians(2)  # omega uncertainty, rad/s
P0 = np.diag([sigma_x**2, sigma_y**2, sigma_xdot **
             2, sigma_ydot**2, sigma_w**2])
ekf = EKF(x0, P0)
dt = 1/1  # measurements taken at 1 Hz

for i in range(datalength):

    # call EKF with nearest neighbor association
    # run EKF propogation and measurement matrix calculation
    ekf.update_predict_matrices(dt)
    ekf.predict(dt)
    ekf.update_measurement_matrices()
    S = copy.deepcopy(ekf.S) # FIXME make sure I don't need a deep copy here
    y_hat = copy.deepcopy(ekf.y_hat) # FIXME or here
    
    # sort measurements into rows of measurement vector pairs to pass into NN
    bearings = dataframes['bearings_clean'].iloc[i, :].dropna()
    ranges = dataframes['ranges_clean'].iloc[i, :].dropna()
    ys = np.array([ranges.to_numpy(),bearings.to_numpy()]).T
    
    # find NN measurement
    yNN = NN.findNN(ys, y_hat, S)
    
    # run EKF measurement/correct step with yNN
    ekf.measurement_correct(yNN)
    xs[i,:] = ekf.x_hat
    
    
    
    # call PDAF


# plot results for cluttered sensor data with missed detections


# plot estimated xy plane track trajectory for positive y
fig, ax = plt.subplots()
ax.plot(dataframes['truth'].iloc[:, 0],
        dataframes['truth'].iloc[:, 2], 'r', label='truth')
ax.plot(xs[:,0], xs[:,1], 'b', label='clean NN EKF')
ax.set(xlabel = 'x, m', ylabel = 'y, m',
       title = 'xy plane track trajectory')
ax.legend()
plt.grid(True)

# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
# Plot the true and estimated turn-rate, 𝜔, vs. time
