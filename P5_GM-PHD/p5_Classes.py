# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:32:22 2023

@author: joewe
"""

import numpy as np
import math
import copy
import scipy.linalg as la
import scipy.stats as stats

class Gaussian():
    def __init__(self, weight, mean, covar):
        self.w = weight
        self.m = mean
        self.P = covar
        
class EKF:
    # EKF class with equations derived for range-bearing sensors tracking 
    # targets undergoing coordinated turns w/unkown constant turn rates. 

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
    def predict(self, x_hat, p_hat, dt):
       
        # breakout current state variables for ease of use in following matrix formations
        xdot = x_hat[2]
        ydot = x_hat[3]
        w = x_hat[4]
        
        # F: discrete-time linearized state matrix
        if w == 0: # constant velocity, non-turning model
            F = np.eye(5)
            F[0,2] = dt
            F[1,3] = dt
        else:
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
        
        # merging this function fixed a bug- investigate why later
    #def predict(self, x_hat, p_hat, dt):
        # receives x_hat, p_hat, dt and propogates. Returns apriori x_hat, p_hat, y_hat
        #self.update_predict_matrices(self, x_hat, dt)
        
        # update P_hat to apriori covariance estimate P_k|k-1
        # note: consider joseph form of covariance update equation because covariance is always symmetric positive definite by definition
        p_hat = self.F @ p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        
        # update predicted apriori state with nonlinear propogation equation
        F_nonlinear = copy.deepcopy(self.F) 
        F_nonlinear[:4,4] = 0 # nonlinear update equation is linearized F without omega derivates in column 5
        x_hat = F_nonlinear @ x_hat
        
        return x_hat, p_hat
    
    def nonlinear_measurement(self, x_hat):
        # convert estimated state to expected measurement
       y_hat = np.array([math.sqrt(x_hat[0]**2 + x_hat[1]**2), math.atan2(x_hat[0],x_hat[1])])
       return y_hat
      
        
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
        
        # measurement noise gain matrix M
        self.M = np.eye(2)
        
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
        
    def measurement_correct(self, y_measured, x_hat, p_hat):
        #update measurement matrices
        self.update_measurement_matrices(x_hat, p_hat)
        
        # compare expected measurement to sensor measurement
        y_hat = self.nonlinear_measurement(x_hat)
        innovation = y_measured - y_hat
       
        # posteriori mean state estimate (from apriori state estimate)
        x_hat = x_hat + self.K @ innovation
        
        # update apriori covariance to posteriori covariance 
        p_hat = p_hat - self.K @ self.S @ self.K.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        
        return x_hat, p_hat
        
        
class GMPHD():
    def __init__(self, gaussians):
        # input: gaussians is a list of Guassian objects which will form the 
        # initial Guassian mixture
    
        # init EKF to use for propogation and correction of individual Guassians
        self.KF = EKF()
        
        # define statistics for GM-PHD
        self.Pd = 0.98 # probability of detection
        self.Ps = 0.99 # probability of survival
        self.kappa = 0.0032 # uniform clutter PHD density
        self.PHD = gaussians
        self.prune_threshold = 10E-5
        self.merge_threshold = 4
        self.max_terms = 100
        
        # init cardinality estimate
        self.N = 0
        for element in gaussians:
            self.N += element.w
        
    def run(self, dt, ranges, bearings):
        # input:
        #   dt: time step
        #   ranges: data series of ranges for time step. Each column is a detection
        #   bearings: data series of bearings. Each  column is a detection
        # output:
        #   state vector at current time step
        
        # define ys with rows of measurement vectors 
        ys = np.array([ranges.to_numpy(),bearings.to_numpy()]).T
        
        self.propogate(dt) # gets birthGM, propogation (for w=0), and weight adjustment correct
        self.correct(ys) # working unclear? Maybe?
        self.prune() # works
        self.merge() # not working: deletes element number 1 (big problem dude)
        self.cap() # assume working (doesn't ruin PHD)
        return self.extract_states()
        
    def propogate(self, dt):
        # propogate existing guassian elements
        for element in self.PHD:
            element.w *= self.Ps
            [element.m, element.P] = self.KF.predict(element.m, element.P, dt)
            
        # add birth gaussian elements
        # *could define this in init for greater efficiency
        # *but putting here allows flexibility if birth model becomes non-constant
        birthGM = [
                    Gaussian(0.02, np.array([-1500, 250, 0, 0, 0]), np.diag([2500, 2500, 2500, 2500, 0.0018])),
                   Gaussian(0.02, np.array([-250, 1000, 0, 0, 0]), np.diag([2500, 2500, 2500, 2500, 0.0018])),
                   Gaussian(0.03, np.array([250, 750, 0, 0, 0]), np.diag([2500, 2500, 2500, 2500, 0.0018])),
                   Gaussian(0.03, np.array([1000, 1500, 0, 0, 0]), np.diag([2500, 2500, 2500, 2500, 0.0018]))
                   ]
        self.PHD += birthGM
        
        # update cardinality estimate
        total_birth_weight = 0 
        for gaussian in birthGM:
            total_birth_weight += gaussian.w
        self.N = self.N*self.Ps + total_birth_weight

    def correct(self, ys):
        # ys_measured is a 2d array with rows of measurement vectors
        
        # create new gaussian elements to add based on measurements
        newGM = []
        sum_new_weights = 0
        for y in ys:
            # calculate likelihood this y is detection for each gaussian element
            likelihoods = np.zeros(len(self.PHD))
            i = 0
            for element in self.PHD:
                self.KF.update_measurement_matrices(element.m, element.P)
                y_hat = self.KF.nonlinear_measurement(element.m)
                S = self.KF.S
                measurement_gaussian = stats.multivariate_normal(mean=y_hat, cov=S)
                likelihoods[i] = self.Pd*element.w
                likelihoods[i] *= measurement_gaussian.pdf(y)
                i += 1
            
            # create new Gaussian elements based on measurement likelihoods
            i = 0
            for element in self.PHD:
                new_w = likelihoods[i] / (self.kappa + sum(likelihoods))
                [new_m, new_P] = self.KF.measurement_correct(y, element.m, element.P)
                new_element = Gaussian(new_w, new_m, new_P)
                newGM.append(new_element)
                sum_new_weights += new_w
                i += 1
                
        # correct PHD apriori elements for probability of detection
        for element in self.PHD:
            element.w *= 1-self.Pd
            
        # add new elements to PHD
        self.PHD += newGM
            
        # update cardinality 
        self.N = self.N*(1-self.Pd) + sum_new_weights
    
    def prune(self):
        i = 0
        for element in self.PHD:
            if element.w < self.prune_threshold:
                self.PHD[i] = None
            i += 1
        self.PHD = [element for element in self.PHD if element is not None]
        
    def merge(self):
        # PHD must be sorted by weight such that PHD[0] is highest weight; 
        self.PHD = sorted(self.PHD, key=lambda x: -x.w)
        
        # algorithm fills temp list with newly merged and left-alone terms
        new_phd = []
        self.merge0term(copy.deepcopy(self.PHD), new_phd)
        self.PHD = new_phd
        
    def cap(self):
        # chatGPT recommendation here: learn about how lambdas work later
        self.PHD = sorted(self.PHD, key=lambda x: -x.w)
        self.PHD = self.PHD[:100]
            
    def mahalanobis(self, x1, x2, P1):
        return (x1-x2).T @ np.linalg.inv(P1) @ (x1-x2)
        
    def merge0term(self, phd, new_phd):
        # find terms close enough to merge
        current_element = phd.pop(0) # pop first element to compare the rest to
        close_elements = [current_element]
        indices = []
        
        # find close terms
        i = 0
        for element in phd: # phd is now all terms except what was PHD[0]
            if (self.mahalanobis(element.m, current_element.m, current_element.P)) < self.merge_threshold: # FIXME check this is the correct P
                indices.append(i)
            i += 1
        # add close terms to close elements list
        for index in indices:
            close_elements.append(phd[index])
        # remove close terms from phd
        phd = [val for indx, val in enumerate(phd) if indx not in indices]
        
        if len(close_elements) > 1: # FIXME check this should be 1
            merged_w = 0
            for element in close_elements:
                merged_w += element.w    
                
            merged_m = 0
            for element in close_elements:
                merged_m += element.w*element.m
            merged_m /= merged_w
            
            merged_P = 0 
            for element in close_elements:
                merged_P = element.w*(element.P + (merged_m - element.m)@(merged_m - element.m).T)
            merged_P /= merged_w
            merged_P = 0.5*(merged_P + merged_P.T)
            
            merged_element = Gaussian(merged_w, merged_m, merged_P)
            new_phd.append(merged_element)
            
        else: # add current element to temp # FIXME is this else necessary?
            new_phd.append(current_element)
        
        if len(phd)>0:
            self.merge0term(phd, new_phd)

    def extract_states(self):
        # expects PHD sorted in order of weight; call cap before this 
        output = []
        for element in self.PHD:
            if element.w > 0.5:
                w_round = round(element.w)
                for i in range(0,w_round):
                    output.append(element.m)
            else:
                break # since list is sorted, all elements to follow also are below threshold
        return output
                
                
      