This project simulates the error between two clocks governed by internal parameters phase offset and frequency. The two clocks are simulated both running together nominally, and with a markov jump in time of the slave clock. 
For both cases, the error between the two clock times is estimated using a linear kalman filters. This is done for three process uncertainties, and the results are plotted to compare kalman filter performance with differing 
process noise estimates. 

This repository contains my P1.m file, with the problem solved in matlab. All that is necessary to view the results is clone this repository and run one the Project1.m script in MATLAB. 
