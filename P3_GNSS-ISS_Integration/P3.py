# -*- coding: utf-8 -*-
"""
GNSS/INS Loose Integration script
(a.k.a. Position domain integration)

AEM667 - Navigation and Target Tracking
Project 3

Written by Joey Westermeyer 2023
"""
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from P3_Classes import EKF
import navpy
from queue import Queue
import copy

# import gps and imu data into queue data structure containing panda series for each time step
all_gps_data = pd.read_csv('project3_resources/gps.txt')
all_gps_data.columns = ['time','lat','long','alt','v_N','v_E','v_D']
all_imu_data = pd.read_csv('project3_resources/imu.txt')
all_imu_data.columns = ['time','w_x','w_y','w_z','f_x','f_y','f_z']
ImuQueue = Queue(maxsize = all_imu_data.shape[0])
for i, row in all_imu_data.iterrows():
    ImuQueue.put(row)
GpsQueue = Queue(maxsize = all_gps_data.shape[0])
for i, row in all_gps_data.iterrows():
    GpsQueue.put(row)
    
# initialize kalman filter with initial measurement
gps = GpsQueue.get()
imu = ImuQueue.get()
gps_measurement = np.array([gps['lat'],gps['long'],gps['alt'],gps['v_N'],gps['v_E'],gps['v_D']])
x0 = np.zeros(15)
x0[0:3] = np.array([gps['lat'], gps['long'], gps['alt']])
x0_forPlotting = x0[0:3]
x0[3:6] = np.array([gps['v_N'], gps['v_E'], gps['v_D']])
x0[6:9] = np.zeros(3) # rpy
x0[9:12] = np.array([0.25, 0.077, -0.12]) # accelerometer biases, m/s^2
x0[12:15] = 1e-4*np.array([2.4, -1.3, 5.6]) # gyroscope biases, rad/s
P0 = 0.1 * np.eye(15)
P0[:6, :6] = 1 * np.eye(6)
dt = 0.02
print(x0)
filter = EKF(x0,P0)
print(x0)
print(x0_forPlotting)

# run EKF for the entirety of the gps/imu datasets given 
x_est = np.zeros((GpsQueue.qsize(),15))
p_est = np.zeros((GpsQueue.qsize(),15,15))
times = np.zeros(GpsQueue.qsize())
i = -1
while True:
    try:
        i += 1
        
        # store prior gps measurement to determine if GPS data has updated via comparison
        gps_last = gps_measurement
        
        # pull next time step's gps and imu report
        if GpsQueue.empty():
            break
        gps = GpsQueue.get()
        imu = ImuQueue.get()
        
        # reformat gps and imu data into arrays for input to EKF
        f_b = np.array([imu['f_x'], imu['f_y'], imu['f_z']])
        wb_IB = np.array([imu['w_x'], imu['w_y'], imu['w_z']])
        gps_measurement = np.array([gps['lat'],gps['long'],gps['alt'],gps['v_N'],gps['v_E'],gps['v_D']])
        times[i] = gps['time']
        
        # propogate EKF forward in time with INS update 
        filter.time_update(f_b, wb_IB, dt)
        
        # if new gps data, correct INS estimate according to ES-EKF equations
        if ( abs(gps_measurement[2] - gps_last[2]) > 0.0000001):
            filter.measurement_update(gps_measurement) 
            print(i)
    
        # store estimated vehicle state at each timestep for plotting
        x_est[i] = filter.x_hat
        p_est[i] = filter.p_hat
    except Exception as e:
        print(f'loop broke at {str(i)} with error {str(e)}')
    
    
# convert position to NED for plotting
i = 0
x_ref =  x_est[0][0], x_est[0][1], x_est[0][2]
for row in x_est:
    x_NED = navpy.lla2ned(row[0], row[1], row[2], x_ref[0], x_ref[1], x_ref[2])
    x_est[i][0:3] = x_NED
    i += 1

#%%
# plot results
# x_N vs x_E, inintial point as reference
fig, ax = plt.subplots()
ax.plot(x_est[:,1],x_est[:,0])
ax.set(xlabel = 'East, m', ylabel = 'North, m',
       title = 'North-East coordinates, origin at initial position')
plt.grid(True)

# Altitude vs time history
fig, ax = plt.subplots()
ax.plot(times - times[0], -x_est[:,2])
ax.set(xlabel = 'time, s', ylabel = 'altitude, m',
       title = 'Altitude time history')
plt.grid(True)

# v vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,3], 'r', label='v_N')
ax.plot(times - times[0], x_est[:,4], 'g', label='v_E')
ax.plot(times - times[0], x_est[:,5], 'b', label='v_D')
ax.set(xlabel = 'time, s', ylabel = 'velocity, m/s',
       title = 'navigation frame velocity time history')
plt.grid(True)
ax.legend()

# roll vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,6], 'r', label='roll')
ax.plot(times - times[0], x_est[:,7], 'g', label='pitch')
ax.plot(times - times[0], x_est[:,8], 'b', label='yaw')
ax.set(xlabel = 'time, s', ylabel = 'Euler angle, rad',
       title = 'Euler angle time history')
plt.grid(True)
ax.legend()

# accelerometer x_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,9], 'r', label='b_x')
ax.plot(times - times[0], x_est[:,10], 'g', label='b_y')
ax.plot(times - times[0], x_est[:,11], 'b', label='b_z')
ax.set(xlabel = 'time, s', ylabel = 'accelerometer bias, m/s^2',
       title = 'acclerometer body-frame biases time history')
plt.grid(True)
ax.legend()

# gyroscope x_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,12], 'r', label='b_x')
ax.plot(times - times[0], x_est[:,13], 'g', label='b_y')
ax.plot(times - times[0], x_est[:,14], 'b', label='b_z')
ax.set(xlabel = 'time, s', ylabel = 'gyroscope bias, rad/s',
       title = 'gyroscope body-frame biases time history')
plt.grid(True)
ax.legend()



# =============================================================================
# Check VDOP and HDOP
# =============================================================================

vdops = np.array([])
hdops = np.array([])
for index, mat in enumerate(p_est):
    if mat is not None:
        vdop = np.sqrt(mat[2, 2] * 2)
        hdop = np.sqrt(mat[0, 0] * 2 + mat[1, 1] * 2)
    else:
        vdop = 0
        hdop = 0
    vdops = np.append(vdops, vdop)
    hdops = np.append(hdops, hdop)

# plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Dilution of Precsion")
ax1.plot(vdops)
ax1.set(xlabel="Time (s)")
ax1.set(ylabel="VDOP (m)")

ax2.plot(hdops)
ax2.set(xlabel="Time (s)")
ax2.set(ylabel="HDOP (m)")

