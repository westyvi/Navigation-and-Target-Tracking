This project simulates the error between two clocks governed by internal parameters phase offset and frequency. The two clocks are simulated both running together nominally, and with a markov jump in time of the slave clock. 
For both cases, the error between the two clock times is estimated using a linear kalman filters. This is done for three process uncertainties, and the results are plotted to compare kalman filter performance with differing 
process noise estimates. 
