# -*- coding: utf-8 -*-
"""
GNSS/INS Loose Integration script
AEM667 - Navigation and Target Tracking
Project 3

Written by Joey Westermeyer 2023
"""
import numpy as np
from numpy import linalg as LA
import pandas as pd
import math
from matplotlib import pyplot as plt
from P3_Classes import EKF, DataQueue
import navpy

# import gps and imu data into queue data structure containing panda series for each time step
all_gps_data = pd.read_csv('project3_resources/gps.txt')
all_gps_data.columns = ['time','lat','long','alt','v_N','v_E','v_D']
all_imu_data = pd.read_csv('project3_resources/imu.txt')
all_imu_data.columns = ['time','w_x','w_y','w_z','f_x','f_y','f_z']
ImuQueue = DataQueue(all_imu_data)
GpsQueue = DataQueue(all_gps_data)

# initialize kalman filter
x0 = np.zeros(15)
gps = GpsQueue.dequeue()
imu = ImuQueue.dequeue()
x0[0:3] = np.array([gps['lat'], gps['long'], gps['alt']])
x0[3:6] = np.array([gps['v_N'], gps['v_E'], gps['v_D']])
x0[6:9] = np.zeros(3) # rpy
x0[9:12] = np.array([0.25, 0.077, -0.12]) # 
x0[12:15] = np.array([2.4, -1.3, 5.6])
P0 = np.eye(15)*10

dt = 0.004
filter = EKF(x0,P0)
f_b = np.array([imu['f_x'], imu['f_y'], imu['f_z']])
wb_IB = np.array([imu['w_x'], imu['w_y'], imu['w_z']])
filter.time_update(f_b, wb_IB, dt)

'''
for i, row in all_gps_data.iterrows():
    pass
'''

        