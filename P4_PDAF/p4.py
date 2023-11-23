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
from p4_Classes import EKF
import os

# Load csv files into dictionary of data frames
resource_path = os.path.join(".", "project4_resources")
filenames = [f for f in os.listdir(resource_path) if f.endswith(".csv")]
dataframes = {}
for filename in filenames:
  basename = os.path.splitext(filename)[0]
  dataframes[basename] = pd.read_csv(os.path.join(resource_path, filename), header=None)
datalength = dataframes['truth'].shape[0]

# intialize target state
xpos = 500 # m
xdot = 7.5 # m/s
ypos = 500 # m
ydot = 7.5 # m/s
w = math.radians(2) # rad/s
x = np.array([xpos, ypos, xdot, ydot, w])
xs = np.array((x.shape[0],datalength))


#for i in range(datalength):
    
    # call EKF with nearest neighbor association
    
    # call PDAF
    
    

# plot results for cluttered sensor data with missed detections


# plot estimated xy plane track trajectory for positive y
# gyroscope x_B bias vs time
fig, ax = plt.subplots()
ax.plot(dataframes['truth'].iloc[:,0], dataframes['truth'].iloc[:,1], 'r', label='b_x')
'''ax.plot(times - times[0], x_est[:,13], 'g', label='b_y')
ax.plot(times - times[0], x_est[:,14], 'b', label='b_z')
ax.set(xlabel = 'time, s', ylabel = 'gyroscope bias, rad/s',
       title = 'gyroscope body-frame biases time history')
ax.legend()'''
plt.grid(True)

# Plot the transformed range and bearing measurements to the position domain overlaid with the true and estimated 𝑥 and 𝑦 position vs. time
# Plot the true and estimated 𝑥 and 𝑦 velocity vs. time
# Plot the true and estimated turn-rate, 𝜔, vs. time
    
    
    
    
    
    
    
    
    
    
    
    