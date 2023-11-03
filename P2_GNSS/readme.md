This project estimates the position and both vertical and horizontal dilution of precisions (VDOP and HDOP) of a rover relative to a stationary base using real, recorded GNSS data. The code uses the recorded ephemeris data 
to calculate space vehicle orbital positions, assumes the base location is the average of estimated positions reported by the base receiver, and uses the double-differenced code-phase psuedorange equations to estimate
the relative position between the base and rover receivers. Nonlinear least squares using the Gauss-Newton algorithm is used to solve these equations. 

To run this project, clone all files in the P2_GNSS folder and run the P2.py file. 
