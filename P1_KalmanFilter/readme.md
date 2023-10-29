This project simulates the error between two clocks governed by internal parameters phase offset and frequency. The two clocks are simulated both running together nominally, and with a markov jump in time of the slave clock. 
For both cases, the error between the two clock times is estimated using a linear kalman filters. This is done for three process uncertainties, and the results are plotted to compare kalman filter performance with differing 
process noise estimates. 

This repository contains my P1.m file, with the problem solved in matlab and Aabhash Bandari's P1.py file which solves the problem in Python. All that is necessary to view the results of either is clone this repository and run one of these two scripts in its respective environment. 
