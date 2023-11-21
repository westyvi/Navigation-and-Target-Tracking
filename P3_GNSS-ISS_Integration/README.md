# AEM667

This project implements a loose INS/GNSS sensor fusion algorithm to output position, attitude, and velocity. This implementation uses an error-state EKF to propogate the highly nonlinear attitude update equations in the INS update step. The fused output trajectory is graphed, along with other pertinent filter output. 

Todos: 
- overlay raw gps data on fused track
- plot apriori estimate immediately before correction to view GPS correction effect
- learn how matplotlib connects plotted points - splines? Straight lines? 
- investigate dilution of precision calculations further
- investigate attitude determination 