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
from pathlib import Path
import re

all_gsp_data = pd.read_table('project3_resources/gps.txt')
all_imu_data = pd.read_table('project3_resources/imu.txt')

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
        