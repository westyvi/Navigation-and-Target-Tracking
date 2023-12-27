# -*- coding: utf-8 -*-
"""
GNSS processing script
AEM667 - Navigation and Target Tracking
Project 2

Written by Joey Westermeyer 2023
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd
import math
from matplotlib import pyplot as plt
from project2_resources import gps_data_parser as gps
from pathlib import Path
import re


# section 1: compute average ECEF base receiver position from SuperStar II
base_ECEF_data = pd.read_table('project2_resources/data_base/ecef_rx0.txt')
base_ECEF_data = base_ECEF_data.drop(base_ECEF_data.columns[range(8,18)], axis = 1)
base_ECEF_data.columns = ['GPS_time','GPS_ week','X','Y','Z','Vx','Vy','Vz']
base_mean_ECEF = np.array([base_ECEF_data['X'].mean(), base_ECEF_data['Y'].mean(),
                 base_ECEF_data['Z'].mean()])
print("Mean base location, ECEF (m): " )
print(base_mean_ECEF)
print('\n')
print("lat Long: ")
base_mean_lla = gps.ecef_to_lla(base_mean_ECEF)
base_mean_lla= [math.degrees(base_mean_lla[0]), math.degrees(base_mean_lla[1]), base_mean_lla[2]]
print(base_mean_lla) # yields answer different to online conversions of xyz to lla
print('(Row the boat Ski-U-Mah go gophers)')

# 2: compute LOS vectors from base to all GPS SV's
nav_msgs = gps.parse_rinex_v2('project2_resources/brdc2930.11n')

# parse base SV icp's
bsats = dict()
directory = 'project2_resources/data_base'
files = Path(directory).glob('icp_sat*')
times = np.array(0)
for file in files:
    satNum = re.search('icp_sat(.*).txt', file.__str__())
    satNum = satNum.group(1)
    (pranges, trans_times, phases, cycle_slips), times = gps.parse_icp(file, times)
    bsats[satNum] = {'pranges':pranges, 'trans_times':trans_times, 'phases':phases, 'slips':cycle_slips}
times = np.delete(times,0) # times array is now times from all base files
    
# parse rover SV icp's
rsats = dict()
directory = 'project2_resources/data_rover'
files = Path(directory).glob('icp_sat*')
for file in files:
    satNum = re.search('icp_sat(.*).txt', file.__str__())
    satNum = satNum.group(1)
    (pranges, trans_times, phases, cycle_slips), times = gps.parse_icp(file, times)
    rsats[satNum] = {'pranges':pranges, 'trans_times':trans_times, 'phases':phases, 'slips':cycle_slips}
    # times array is now times from all base AND rover  files

# calculate base LOS to each SV (NED coords) for each time step
for sat in bsats:
    LOS_ECEF = nav_msgs[sat].calculate_orbit(times)
    i = 0
    LOS_NED = np.zeros((times.size, 3))
    for row in LOS_ECEF:
        LOS_NED[i][:] = gps.ecef_to_ned(row, base_mean_ECEF)
        i += 1
    bsats[sat]['LOS'] = LOS_NED

# find highest satellite, set as reference
base_ref = np.empty(times.size)
i = 0
for t in times:
    tan_theta_max = 0
    for sat in bsats:
        D = bsats[sat]['LOS'][i][2]
        E = bsats[sat]['LOS'][i][1]
        N = bsats[sat]['LOS'][i][0]
        tan_theta = -D/math.sqrt(N**2 + E**2)
        if tan_theta > tan_theta_max:
            tan_theta_max = tan_theta
            base_ref[i] = int(sat)
    i += 1
    
# for each time step, if rover and base have enough available SV's, calculate
# beta by applying NLS with GNA to the nonlinear observation equations 
# if not enough SV's available, populate beta with zeros
i = 0
numrsats = np.empty(times.size)
numbsats = np.empty(times.size)
betas = np.zeros((times.size, 3))
hdop = np.zeros(times.size)
vdop = np.zeros(times.size)
for t in times:
    # create dict of SV's available to the base at time step
    bsat_available = dict()
    for sat in bsats:
        temp = abs(bsats[sat]['trans_times'] - t)
        t_index = temp.argmin()
        t_min = temp[t_index]
        if t_min < 0.2:
            bsat_available[sat] = {'bprange':bsats[sat]['pranges'][t_index], 
                                   'X_NED':bsats[sat]['LOS'][t_index]}
    
    # create dict of SV's available to the rover at time step
    rsat_available = dict()
    for sat in bsats:
        temp = abs(rsats[sat]['trans_times'] - t)
        t_index = temp.argmin()
        t_min = temp[t_index]
        if t_min < 0.2:
            rsat_available[sat] = {'rprange':rsats[sat]['pranges'][t_index]}

    # log num sats avaible to each Rx for viewing/debugging pleasure
    numrsats[i] = len(rsat_available)
    numbsats[i] = len(bsat_available)
    
    # add SV with current time step data to common sats_available dict if both
    # rover and base see it
    sats_available = dict()
    for bkey in bsat_available.keys():
        if bkey in rsat_available.keys():
            sats_available[bkey] = bsat_available[bkey]
    for rkey in rsat_available.keys():
        if rkey in bsat_available.keys():
            sats_available[rkey]['rprange'] = rsat_available[rkey]['rprange']
    
    # if enough available satellites, calculate solution using NLS with GNA
    # if not enough satellites to position, populate beta with zeros
    beta = np.zeros(3)
    if len(sats_available) > 3:
        # init nonlinear observation equation arrays
        refsat = sats_available.pop(str(int(base_ref[i])))
        y = np.empty(len(sats_available))
        f_xb = np.empty(len(sats_available))
        J = np.zeros((y.size, 3))
        NLS_iter = 0
        
        # run NLS GNA on nonlinear observation equations
        # dd: jR - jB - refR + refB
        # where j is jth SV, ref is ref SV, R is rover, B is base
        while NLS_iter < 6:
            j = 0
            # populate observation equation arrays
            for sat in sats_available:
                y[j] = (sats_available[sat]['rprange'] - sats_available[sat]['bprange']
                        - refsat['rprange'] + refsat['bprange'])
                f_xb[j] = ( (LA.norm(beta - sats_available[sat]['X_NED']) - 
                             LA.norm(sats_available[sat]['X_NED']))
                           - (LA.norm(beta - refsat['X_NED']) 
                              - LA.norm(refsat['X_NED'])))
                J[j] = ( ((beta - sats_available[sat]['X_NED'])
                        /LA.norm(beta - sats_available[sat]['X_NED'])) 
                        - (beta - refsat['X_NED'])/LA.norm(beta - refsat['X_NED']))
                j += 1
                
            # find NLS/GNA step, apply to beta
            J_psuedo = (LA.inv(J.transpose() @ J)) @ J.transpose()
            residual = (y-f_xb)
            delta = J_psuedo @ residual
            beta += delta
            if max(abs(delta/beta)) < 0.001:
                break
            if(NLS_iter) == 5:
                print('max iterations reached in NLS/GNA loop')
            NLS_iter += 1
        J_almost_psuedo = LA.inv(J.transpose() @ J)
        vdop[i] = J_almost_psuedo[2,2]
        hdop[i] = math.sqrt(J_almost_psuedo[0,0]**2 + J_almost_psuedo[1,1]**2)
        betas[i] = beta
    i += 1
    
# plot N-E coords
fig, ax = plt.subplots()
ax.plot(betas[:,1],betas[:,0])
ax.set(xlabel = 'East, m', ylabel = 'North, m',
       title = 'North-East coordinates, base at origin')
plt.grid(True)

# plot altitude vs time
fig, ax = plt.subplots()
ax.plot(times-times[0],-betas[:,2])
ax.set(xlabel = 'time, s', ylabel = 'Altitude, m',
       title = 'Height-time coordinates, base at origin')
plt.grid(True)

# plot hdop vs time
fig, ax = plt.subplots()
ax.plot(times-times[0],hdop)
ax.set(xlabel = 'time, s', ylabel = 'HDOP',
       title = 'Horizontal Dilution of Precision vs time')
plt.grid(True)

# plot vdop vs time
fig, ax = plt.subplots()
ax.plot(times-times[0],vdop)
ax.set(xlabel = 'time, s', ylabel = 'VDOP',
       title = 'Vertical Dilution of Precision vs time')
plt.grid(True)

