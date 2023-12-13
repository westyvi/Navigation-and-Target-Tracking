# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:32:22 2023

@author: joewe
"""

import numpy as np
import math
import copy
from scipy.stats.distributions import chi2
import scipy.linalg as la
import scipy.stats as stats

class EKF:

    def __init__(self):
        
        # init Weiner process model (for CT-UTR) uncertainty parameters
        self.sigma_x = 10 # x uncertainty, m
        self.sigma_y = 10 # y uncertainty, m
        self.sigma_xdot = 5 # xdot uncertainty, m/s
        self.sigma_ydot = 5 # ydot uncertainty, m/s
        self.sigma_w = math.radians(2) # omega uncertainty, rad/s
        
        # play with gains for educational purposes
        '''self.sigma_xdot = .3 # xdot uncertainty, m/s
        self.sigma_ydot = .3 # ydot uncertainty, m/s
        self.sigma_w = math.radians(.5) # omega uncertainty, rad/s
        self.sigma_r = 20
        self.sigma_theta = math.radians(10)'''
        
        # init measurement model uncertainty parameters 
        self.sigma_r = 10 # range uncertainty, m
        self.sigma_theta = math.radians(2) # bearing uncertainty, rad
        
     # define and assign the update matrices 
     # (at the current state estimate x_hat) as class properties
    def update_predict_matrices(self, x_hat, dt):
       
        # breakout current state variables for ease of use in following matrix formations
        xdot = x_hat[2]
        ydot = x_hat[3]
        w = x_hat[4]
        
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
        self.Q = np.diag([self.sigma_xdot**2, self.sigma_ydot**2, self.sigma_w**2])
        
        # measurement noise gain matrix M
        self.M = np.eye(2)
        
    def predict(self, x_hat, p_hat, dt):
        # receives x_hat, p_hat, dt and propogates. Returns apriori x_hat, p_hat, y_hat
        
        # update P_hat to apriori covariance estimate P_k|k-1
        # note: consider joseph form of covariance update equation because covariance is always symmetric positive definite by definition
        p_hat = self.F @ p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        
        # update predicted apriori state with nonlinear propogation equation
        F_nonlinear = copy.deepcopy(self.F) 
        F_nonlinear[:4,4] = 0 # nonlinear update equation is linearized F without omega derivates in column 5
        x_hat = F_nonlinear @ self.x_hat
        
        # convert estimated state to expected measurement
        y_hat = np.array([math.sqrt(x_hat[0]**2 + x_hat[1]**2), math.atan2(x_hat[0],x_hat[1])])
        
        return x_hat, p_hat, y_hat
    
    def update_measurement_matrices(self, x_hat, p_hat):
        # updates instance's measurement equations (as instance vals) for input x_hat, p_hat
        
        # breakout current state variables for ease of use in following matrix formations
        x = x_hat[0]
        y = x_hat[1]
        
        # measurement noise covariance matrix
        self.R = np.diag([self.sigma_r**2, self.sigma_theta**2])

        # define linearized output matrix H
        self.H = np.array([
            [x/math.sqrt(x**2 + y**2), y/math.sqrt(x**2 + y**2), 0, 0, 0],
            [1/(y + x**2/y), -x/(x**2 + y**2), 0, 0, 0]
                ])
        
        # innovation covariance noise covariance matrix S
        self.S = self.H @ p_hat @ self.H.T + self.M @ self.R @ self.M.T
        
        # enforece symmetry 
        self.S = 0.5*(self.S + self.S.T)
        
        # enforce positive semidefinite
        self.S = la.sqrtm(self.S.T @ self.S)
        _ = np.linalg.pinv(self.S)
        self.S = np.linalg.pinv(_)
       
        # Kalman gain matrix
        self.K = p_hat @ self.H.T @ np.linalg.inv(self.S)
        
    def measurement_correct(self, y_measured, x_hat, p_hat, y_hat):
        # compare expected measurement to sensor measurement
        innovation = y_measured - y_hat
       
        # posteriori mean state estimate (from apriori state estimate)
        x_hat = self.x_hat + self.K @ innovation
        
        # update apriori covariance to posteriori covariance 
        p_hat = self.p_hat - self.K @ self.S @ self.K.T
        p_hat = 0.5*(self.p_hat + self.p_hat.T) # enforce symmetry
        
        return x_hat, p_hat
        
        
class GMPHD(EKF):
    def __init__(self, x0, P0):
        super().__init__() 
        
        # define statistics for GM-PHD
        self.Pd = 0.98 # probability of detection
        self.Ps = 0.99 # probability of survival
        self.kappa = 0.0032 # uniform clutter PHD density
        
    def run(self, dt, ranges, bearings):
        # input:
        #   dt: time step
        #   ranges: data series of ranges for time step. Each column is a detection
        #   bearings: data series of bearings. Each  column is a detection
        # output:
        #   state vector at current time step
        
        # call EKF with nearest neighbor association
        # run EKF propogation and measurement matrix calculation
        self.update_predict_matrices(dt)
        self.predict(dt)
        self.update_measurement_matrices()
        
        # sort measurements into rows of measurement vector pairs to pass into NN
        ys = np.array([ranges.to_numpy(),bearings.to_numpy()]).T
        
        # find measurements inside gate
        ys_gated = self.gate(ys)
        
        # if >= 1 gated measurement, run measurement update
        if ys_gated.shape[0] >= 1:
            self.measurement_correct(ys_gated)
        else:
            #print('pdaf missed')
            pass
            
        return self.x_hat
        
    def mahalanobis(x1, x2, P1):
        return (x1-x2).T @ np.linalg.inv(P1) @ (x1-x2)
             
    def measurement_correct(self, ys_measured):
        # ys_measured is already gated
        # ys_measured is a 2d array with rows of measurement vectors
        # if no ys_measured in gate, skip measurement_correct step
        
        # calculate likelihoods of all gated measurements
        data_association_probabilities = np.zeros(ys_measured.shape[0])
        likelihoods = np.zeros(ys_measured.shape[0])
        i = 0
        c = 1/self.gamma_c
        measurement_gaussian = stats.multivariate_normal(mean=self.y_hat, cov=self.S)
        for y in ys_measured:
            likelihoods[i] = c*self.Pd*measurement_gaussian.pdf(y) # FIXME check this
            i += 1
            
        # sum all likelihoods within the gate
        total_likelihood = np.sum(likelihoods)
        
        # calculate data association probabilities
        i = 0
        for y in ys_measured:
            num = likelihoods[i]
            den = 1 - self.Pd*self.Pg + total_likelihood
            data_association_probabilities[i] = num/den
            i += 1
        no_target_probability = (1-self.Pd*self.Pg)/(1-self.Pd*self.Pg+total_likelihood)
        
        # calculate probability-averaged innovation
        innovation = 0
        i = 0
        for prob in data_association_probabilities:
            innovation += prob*(ys_measured[i] - self.y_hat)
        
        # posteriori mean state estimate
        self.x_hat = self.x_hat + self.K @ innovation
        
        # calculate uncertainty in which measurement is correct for covariance update
        measurement_uncertainty_sum = 0
        i = 0
        for prob in data_association_probabilities:
            # investigate: making this dot product instead of outer product causes 
            # ignorance of measurements and a perfectly circular path. Why?
            measurement_uncertainty_sum += (
                np.outer((prob*(ys_measured[i] - self.y_hat)), (ys_measured[i] - self.y_hat)))
            i += 1
        measurement_uncertainty_sum -= np.outer(innovation, innovation.T)
        
        # update apriori covariance to posteriori covariance 
        self.p_hat = ((1 - no_target_probability)*(self.p_hat - self.K @ self.S @ self.K.T)
                    + no_target_probability * self.p_hat
                    + self.K @ (measurement_uncertainty_sum) @ self.K.T)
        
        # enforce symmetry
        self.p_hat = 0.5*(self.p_hat + self.p_hat.T) 
