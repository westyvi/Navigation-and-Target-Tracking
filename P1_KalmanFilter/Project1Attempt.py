# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:29:35 2023

@author: Joey Westermeyer
"""

import numpy as np
import math

# establish x0 as the first column of state vector history x
xhistory = np.array([ [0.25], [math.pi/4] ])

# set constant variables
T = 0.01
r = 0.05
F = np.array([ [1, T], [0, 1] ])

# prove I can multiply matrices
A = np.array([[1,0],[1,1]])
B = np.array([[1,2],[3,4]])
C = A@B
print(C)

# log the index of the state array x
k = 0

for t in np.arange(0,30,T):
    x = F@xhistory[k,0]
    xhistory = np.append(xhistory,np.array([[t], [0]]))
    

print(xhistory)

