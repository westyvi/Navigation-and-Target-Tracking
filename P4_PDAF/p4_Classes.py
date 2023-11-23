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
        
    # propogate forward in time with only a proprioceptive (IMU) measurement:
    # updates nominal state and error state covariance (error state remains zero)
    def update(self, y_measured, dt):
       
        # breakout current state variables for ease of use in following matrix formations
        w = self.x_hat[4]
        xdot = self.x_hat[2] # FIXME check order of state vars
        ydot = self.x_hat[3]
        x = self.x_hat[0]
        y = self.x_hat[1]
        
        # ------- define EKF propogtion matrices: -----------------------------------------
        # F: discrete-time state matrix
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
        
        # L: process noise gain matrix
        L = np.zeros((5,3))
        L[0,0] = dt**2/2
        L[1,1] = dt**2/2
        L[2,0] = dt
        L[3,1] = dt
        L[4,2] = 1
        
        # process noise w covariance matrix
        Q = np.diag([self.sigma_x**2, self.sigma_y**2, self.sigma_xdot**2, self.sigma_ydot**2, self.sigma_w**2])
        
        # measurement noise covariance matrix
        R = np.diag([self.sigma_r**2, self.sigma_theta**2])

        # define linearized output matrix H
        H = np.array([
            [x/math.sqrt(x**2 + y**2), y/math.sqrt(x**2 + y**2)],
            [1/(y + x**2/y), -1/(x**2 + x**4/y**2)]
                ])
        
        # measurement noise gain matrix M
        M = np.eye(2)
        
        # ------- update equations: --------------------------------------------
        # update P_hat to apriori covariance estimate P_k|k-1
        # consider joseph form of covariance update equation because covariance is always symmetric positive definite by definition
        self.p_hat = F @ self.p_hat @ F.T + L @ Q @ L.T
        self.p_hat = 0.5*(self.p_hat + self.p_hat.T) # enforce symmetry
        
        # update predicted apriori state with nonlinear propogation equation
        different_F = copy.deepcopy(F)
        different_F[:4,4] = 0 # unsure if this is the write nonlinear update equation, but its 18.51 in the book
        self.x_hat = different_F @ self.x_hat
        
        # ------- correct/measurement equations: --------------------------------------------
        # convert estimated state to expected measurement, compare to sensor measurement
        y_hat = np.array([math.sqrt(x**2 + y**2), math.atan2(x,y)])
        innovation = y_measured - y_hat
       
        # innovation covariance noise covariance matrix S
        S = H @ self.p_hat @ H.T + M @ R @ M.T
       
        # Kalman gain matrix
        K = self.p_hat @ H.T @ np.linalg.inv(S)
        
        # posteriori mean state estimate (from apriori state estimate)
        self.x_hat = self.x_hat + K @ innovation
        
        # update apriori covariance to posteriori covariance 
        self.p_hat = self.p_hat - K @ S @ K.T
        self.p_hat = 0.5*(self.p_hat + self.p_hat.T) # enforce symmetry
        
        
        
        