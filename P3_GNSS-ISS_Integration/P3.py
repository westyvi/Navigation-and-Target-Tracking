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
from P3_Classes import EKF, DataQueue
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
x0[3:6] = np.array([gps['v_N'], gps['v_E'], gps['v_D']])
x0[6:9] = np.zeros(3) # rpy
x0[9:12] = np.array([0.25, 0.077, -0.12]) # accelerometer biases, m/s^2
x0[12:15] = 1e-4*np.array([2.4, -1.3, 5.6]) # gyroscope biases, rad/s
P0 = np.eye(15)*1
dt = 0.02
filter = EKF(x0,P0)
    
# run EKF for the entirety of the gps/imu datasets given 
x_est = np.zeros((GpsQueue.qsize(),15))
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
    except:
        print('loop broke at')
        print(i)
        break
    
    
# convert position to NED for plotting
i = 0
for row in x_est:
    x_NED = navpy.lla2ned(row[0], row[1], row[2], x_est[0][0], x_est[0][1], x_est[0][2],)
    x_est[i][0:3] = x_NED
    i += 1
    
# plot results
# x_N vs x_E, inintial point as reference
fig, ax = plt.subplots()
ax.plot(x_est[:,1],x_est[:,0])
ax.set(xlabel = 'East, m', ylabel = 'North, m',
       title = 'North-East coordinates, base at origin')
plt.grid(True)

# Altitude vs time history
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,2])
ax.set(xlabel = 'time, s', ylabel = 'altitude, m',
       title = 'Altitude-time history')
plt.grid(True)

# v_N vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,3])
ax.set(xlabel = 'time, s', ylabel = 'north-velocity, m/s',
       title = 'velocity_N-time history')
plt.grid(True)

# v_E vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,4])
ax.set(xlabel = 'time, s', ylabel = 'east-velocity, m/s',
       title = 'velocity_E-time history')
plt.grid(True)

# v_down vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,5])
ax.set(xlabel = 'time, s', ylabel = 'down-velocity, m/s',
       title = 'velocity_D-time history')
plt.grid(True)

# roll vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,6])
ax.set(xlabel = 'time, s', ylabel = 'roll, rad',
       title = 'roll-time history')
plt.grid(True)

# pitch vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,7])
ax.set(xlabel = 'time, s', ylabel = 'pitch, rad',
       title = 'pitch-time history')
plt.grid(True)

# yaw vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,8])
ax.set(xlabel = 'time, s', ylabel = 'yaw, rad',
       title = 'yaw-time history')
plt.grid(True)

# accelerometer x_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,9])
ax.set(xlabel = 'time, s', ylabel = 'accelerometer x_B bias, m/s^2',
       title = 'acclerometer body x_bias-time history')
plt.grid(True)

# accelerometer y_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,10])
ax.set(xlabel = 'time, s', ylabel = 'accelerometer y_B bias, m/s^2',
       title = 'acclerometer body y_bias-time history')
plt.grid(True)

# accelerometer z_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,11])
ax.set(xlabel = 'time, s', ylabel = 'accelerometer z_B bias, m/s^2',
       title = 'acclerometer body z_bias-time history')
plt.grid(True)

# gyroscope x_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,12])
ax.set(xlabel = 'time, s', ylabel = 'gyroscope x_B bias, rad/s',
       title = 'gyroscope body x_bias-time history')
plt.grid(True)

# gyroscope y_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,13])
ax.set(xlabel = 'time, s', ylabel = 'gyroscope y_B bias, rad/s',
       title = 'gyroscope body y_bias-time history')
plt.grid(True)

# gyroscope z_B bias vs time
fig, ax = plt.subplots()
ax.plot(times - times[0], x_est[:,14])
ax.set(xlabel = 'time, s', ylabel = 'gyroscope z_B bias, rad/s',
       title = 'gyroscope body z_bias-time history')
plt.grid(True)

    


        