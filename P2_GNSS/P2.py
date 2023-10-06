# -*- coding: utf-8 -*-
"""
GNSS processing script
AEM667 - Navigation and Target Tracking
Project 2

Written by Joey Westermeyer 2023
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from project2_resources import gps_data_parser as gps

# section 1: compute average ECEF base receiver position from SuperStar II
base_ECEF_data = pd.read_table('project2_resources/data_base/ecef_rx0.txt')
base_ECEF_data = base_ECEF_data.drop(base_ECEF_data.columns[range(8,18)], axis = 1)
base_ECEF_data.columns = ['GPS_time','GP_ week','X','Y','Z','Vx','Vy','Vz']
base_mean_ECEF = np.array([base_ECEF_data['X'].mean(), base_ECEF_data['Y'].mean(),
                 base_ECEF_data['Z'].mean()])
print("Mean base location, ECEF (m): " )
print(base_mean_ECEF)
print('\n')

# 2: compute LOS vectors from base to all GPS SV's
nav_msgs = gps.parse_rinex_v2('project2_resources/brdc2930.11n')


satellites = list()
satNums = [2,3,4,9,10,12,17,23,25]
for satNum in satNums:
    fname = "project2_resources/data_base/icp_sat" + str(satNum) + ".txt"
   # base_icps[0] = gps.parse_icp(fname, list())