# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:32:22 2023

@author: joewe
"""

import numpy as np
import math
import copy


class EKF:

    def __init__(self, x0, P0):
        self.x_hat = x0 # init nominal state, [L,L,A,vn,ve,vd,r,p,y,bax,bay,baz,bgx,bgy,bgz]
        self.p_hat = P0 # init covariance matrix
        
        # init Weiner process model (for CT-UTR) uncertainty parameters
        self.sigma_x = 10 # x uncertainty, m
        self.sigma_y = 10 # y uncertainty, m
        self.sigma_xdot = 5 # xdot uncertainty, m/s
        self.sigma_ydot = 5 # ydot uncertainty, m/s
        self.sigma_w = math.radians(2) # omega uncertainty, rad/s
        
        # init measurement model uncertainty parameters 
        self.sigma_r = 10 # range uncertainty, m
        self.sigma_theta = math.radians(2) # bearing uncertainty, rad
        self.Pd = 0.9 # probability of detection
        self.gamma_c = 0.0032 # Poisson clutter spatial density
        
    def update_predict_matrices(self, dt):
       
        # breakout current state variables for ease of use in following matrix formations
        xdot = self.x_hat[2]
        ydot = self.x_hat[3]
        w = self.x_hat[4]
        
        # ------- define EKF propogtion matrices: -----------------------------------------
        # F: discrete-time linearized state matrix
        F = np.eye(5)
        F[:2, 2:4] = [[math.sin(w*dt)/w, (math.cos(w*dt)-1)/w],
                       [(1-math.cos(w*dt))/w, math.sin(w*dt)/w]]
        F[2:4, 2:4] = [[math.cos(w*dt), -math.sin(w*dt)], [math.sin(w*dt), math.cos(dt*w)]]
        F[0,4] = ( ((w*dt*math.cos(dt*w)-math.sin(w*dt))/(w**2))*xdot
                    - (w*dt*math.sin(w*dt) - 1 + math.cos(w*dt))*ydot/(w**2) )
        F[1,4] = ( ((w*dt*math.sin(dt*w) - 1 + math.cos(w*dt))/(w**2))*xdot
                    + (w*dt*math.cos(w*dt) - math.sin(w*dt))*ydot/(w**2) )
        F[2,4] = -dt*math.sin(w*dt)*xdot - dt*math.cos(w*dt)*ydot
        F[3,4] = dt*math.cos(dt*w)*xdot - dt*math.sin(w*dt)*ydot
        F[4,4] = 1 # beta=1 for weiner process model for omega
        self.F = F
        
        # L: process noise gain matrix
        L = np.zeros((5,3))
        L[0,0] = dt**2/2
        L[1,1] = dt**2/2
        L[2,0] = dt
        L[3,1] = dt
        L[4,2] = 1
        self.L = L
        
        # process noise w covariance matrix
        self.Q = np.diag([self.sigma_x**2, self.sigma_y**2, self.sigma_xdot**2, self.sigma_ydot**2, self.sigma_w**2])
        self.Q = np.diag([self.sigma_xdot**2, self.sigma_ydot**2, self.sigma_w**2])
        
        # measurement noise gain matrix M
        self.M = np.eye(2)
        
    def predict(self, dt):
        # ------- predict/update equations: --------------------------------------------
        # update P_hat to apriori covariance estimate P_k|k-1
        # consider joseph form of covariance update equation because covariance is always symmetric positive definite by definition
        self.p_hat = self.F @ self.p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        self.p_hat = 0.5*(self.p_hat + self.p_hat.T) # enforce symmetry
        
        # update predicted apriori state with nonlinear propogation equation
        F_nonlinear = copy.deepcopy(self.F) # nonlinear update equation is linearized F without omega derivates in column 5
        F_nonlinear[:4,4] = 0
        self.x_hat = F_nonlinear @ self.x_hat
        
        # convert estimated state to expected measurement
        self.y_hat = np.array([math.sqrt(self.x_hat[0]**2 + self.x_hat[1]**2), math.atan2(self.x_hat[0],self.x_hat[1])])
    
    def update_measurement_matrices(self):
        # breakout current state variables for ease of use in following matrix formations
        x = self.x_hat[0]
        y = self.x_hat[1]
        
        # measurement noise covariance matrix
        self.R = np.diag([self.sigma_r**2, self.sigma_theta**2])

        # define linearized output matrix H
        self.H = np.array([
            [x/math.sqrt(x**2 + y**2), y/math.sqrt(x**2 + y**2), 0, 0, 0],
            [1/(y + x**2/y), -1/(x**2 + x**4/y**2), 0, 0, 0]
                ])
        
        # innovation covariance noise covariance matrix S
        self.S = self.H @ self.p_hat @ self.H.T + self.M @ self.R @ self.M.T
       
        # Kalman gain matrix
        self.K = self.p_hat @ self.H.T @ np.linalg.inv(self.S)
        
    def measurement_correct(self, y_measured):
        # ------- correct/measurement equations: --------------------------------------------
        # compare expected measurement to sensor measurement
        innovation = y_measured - self.y_hat
       
        # posteriori mean state estimate (from apriori state estimate)
        self.x_hat = self.x_hat + self.K @ innovation
        
        # update apriori covariance to posteriori covariance 
        self.p_hat = self.p_hat - self.K @ self.S @ self.K.T
        self.p_hat = 0.5*(self.p_hat + self.p_hat.T) # enforce symmetry
        
        
class NN:
    
    def runNNEKF(ekf, dt, ranges, bearings):
        # input:
        #   ekf: already initialized ekf class
        #   dt: time step
        #   ranges: data series of ranges for time step. Each column is a detection
        #   bearings: data series of bearings. Each  column is a detection
        # output:
        #   state vector at current time step
        
        # call EKF with nearest neighbor association
        # run EKF propogation and measurement matrix calculation
        ekf.update_predict_matrices(dt)
        ekf.predict(dt)
        ekf.update_measurement_matrices()
        S = copy.deepcopy(ekf.S) # FIXME make sure I don't need a deep copy here
        y_hat = copy.deepcopy(ekf.y_hat) # FIXME or here
        
        # sort measurements into rows of measurement vector pairs to pass into NN
        ys = np.array([ranges.to_numpy(),bearings.to_numpy()]).T
        
        # find NN measurement
        yNN = NN.findNN(ys, y_hat, S)
        
        # run EKF measurement/correct step with yNN
        ekf.measurement_correct(yNN)
        return ekf.x_hat
        
    def findNN(ys, y_hat, S):
        # returns measurement y that is the nearest neighbor to predicted measurement y_hat
        # by mahalanobis distance
        # inputs:
        #   ys: 2D numpy array. Each row is a measurement detection with columns containing the states
        #   y_hat: 1D numpy array of the expected measurement, columns are state variables just as in y
        #   S: innovation covariance matrix
        # output:
        #   y: 1D numpy array containing measurement vector that is NN to y_hat
        yNN = ys[0]
        mindist = (yNN-y_hat).T @ np.linalg.inv(S) @ (yNN-y_hat)
        for y in ys:
            dist = (y-y_hat).T @ np.linalg.inv(S) @ (y-y_hat)
            if dist < mindist:
                yNN = y
        return yNN
    
                
        
        
        