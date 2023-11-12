# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:32:22 2023

@author: joewe
"""

from queue import Queue
import numpy as np
import math
import navpy
from scipy.linalg import expm

# class converts a pandas datatable with rows of information to a queue of data seriess
class DataQueue:
    def __init__(self, data_table):
        self.queue = Queue(maxsize = data_table.shape[0])
        for i, row in data_table.iterrows():
            self.queue.put(row)

    def enqueue(self, row):
        self.queue.put(row)

    def dequeue(self):
        if not self.queue.empty():
            return self.queue.get()
        else :
            return ValueError('queue empty: cannot dequeue')

    def __str__(self):
        return f'queue empty = {self.queue.empty()}'


class EKF:

    def __init__(self, x0, P0):
        self.x_hat = x0 # init nominal state
        self.dx_hat = x0*0 # init error state as zeros
        self.p_hat = P0 # init covariance matrix
        self.DCM = Euler2DCM(x0[6:9])
        self.imu = IMU()
        
        # init uncertainty parameters for EKF
        self.sigma_p = 3 # GNSS position, m
        self.sigma_v = 0.2 # GNSS velocity, m/s
        self.sigma_ba = 0.0005 # accelerometer Markov bias stddev, g
        self.t_a = 300 # accelerometer bias time const, s
        self.sigma_wa = 0.12 # accelerometer output noise stddev, g
        self.sigma_bg = math.radians(0.3) # gyroscope Markov bias stddev, rad/sqrt(s)
        self.t_g = 300 # gyroscope bias time const, s
        self.sigma_wg = math.radians(0.95) # gyroscope output noise stddev, rad/s
        
    def time_update(self, f_b, wb_IB, dt):
        # propogate forward in time with only a proprioceptive (IMU) measurement:
        # nominal state updates
        # error state covariance updates (error state remains zero)
        
        
        # define variables used in following matrix formations
        a = 6378137
        wn_IE = wn_IECalc(self.x_hat)
        wn_EN = wn_ENCalc(self.x_hat)
        g = gravityCalc(self.x_hat[0], self.x_hat[2])
        
        # define EKF matrices
        A = np.block([
                      [np.block([-navpy.skew(wn_EN), np.eye(3), np.zeros((3,9))])],
                      [np.block([ g/a*np.diag([-1,-1,2]), -navpy.skew(2*wn_IE + wn_EN ), navpy.skew(self.DCM @ f_b), self.DCM, np.zeros((3,3))])],
                      [np.block([np.zeros((3,6)), -navpy.skew(wn_EN + wn_IE), np.zeros((3,3)), -self.DCM])],
                      [np.block([np.zeros((3,9)), -1/self.t_a*np.eye(3), np.zeros((3,3))])],
                      [np.block([np.zeros((3,12)), -1/self.t_g*np.eye(3)])]
                     ])
        L = np.block([
                      [np.zeros((3,12))],
                      [np.block([self.DCM,np.zeros((3,9))])],
                      [np.block([np.zeros((3,3)),-self.DCM,np.zeros((3,6))])],
                      [np.block([np.zeros((3,6)),np.eye(3),np.zeros((3,3))])],
                      [np.block([np.zeros((3,9)),np.eye((3))])]
                    ])
        
        sigma_mua = 2*math.pow(self.sigma_ba,2)/self.t_a
        sigma_mug = 2*math.pow(self.sigma_bg,2)/self.t_g
        Sw = np.block([
                       [np.block([math.pow(self.sigma_wa,2)*np.eye(3), np.zeros((3,9))])],
                       [np.block([np.zeros((3,3)), math.pow(self.sigma_wg,2)*np.eye(3), np.zeros((3,6))])],
                       [np.block([np.zeros((3,6)), math.pow(sigma_mua,2)*np.eye(3), np.zeros((3,3))])],
                       [np.block([np.zeros((3,9)), math.pow(sigma_mug,2)*np.eye(3)])],
                      ])
        F = expm(A*dt)
        Q = ( np.eye(15) + dt*A )@( dt*L@Sw@L.T)

        # update P to apriori covariance estimate P_k|k-1
        self.p_hat = F @ self.p_hat @ F.T + Q
        
        # update nominal state
        print(self.x_hat)
        self.x_hat = self.imu.INSUpdate(self.x_hat, f_b, wb_IB, dt)
        print(self.x_hat)
        

    def measurement_update():
        # overwrite altitude estimate with GNSS solution
        
        # innovation computation
        # something something transfrom INS nominal state p and GNSS p_measured to same NED frame
        # ...with current INS nominal state as [000] ref point of that NED frame
        pass
    
    def state_update(self):
        # update position: must convert dxdydz to dldlda
        dlat = self.dx_hat[0]/( rnCalc(self.x_hat[0]) + self.x_hat[2])
        dlong = self.dx_hat[1]/( (reCalc(self.x_hat[0]) + self.x_hat[2]) * math.cos(self.x_hat[0]) )
        dalt = -self.dx_hat[2]
        self.x_hat[0:3] = self.x_hat[0:3] + np.array([dlat, dlong, dalt])
        g = gravityCalc(self.x_hat[0], self.x_hat[2])
        gN = np.array([0,0,g])
        
        # update velocity
        
        
# IMU class stores current DCM and state
# contains methods to output updated DCM and state given IMU f_B and w_B measurements
class IMU:
    def __init__(self, dt=0.004):
        self.a = 6378137
        self.f = 1/298.257223563
        self.e = math.sqrt(self.f*(2-self.f))
        self.dt = dt
        
    # return x_hat with updated euler angles, p_LLA, v_NED 
    # input: x_hat, accelerometer measurement f_B, gryo measurement xb_IB, propogation time dt
    def INSUpdate(self, x_hat, f_B, wb_IB, dt):
        self.updateIMUInternals(x_hat, dt)
        x_hat[6:9] = self.updateEuler(wb_IB)
        x_hat[3:6] = self.updateVelocity(f_B)
        x_hat[0:3] = self.updatePosition(x_hat[3:6])
        return x_hat
        
    # save current copies of vehicle states in IMU class
    def updateIMUInternals(self, x_hat, dt):
        # update state to current estimate
        self.x_hat = x_hat
        self.DCM = Euler2DCM(x_hat[6:9])
        
        # update trasnport rates
        self.wn_IE = wn_IECalc(x_hat)
        self.wn_EN = wn_ENCalc(x_hat)
        self.wn_IN = self.wn_IE + self.wn_EN
        self.wb_IN = self.DCM @ self.wn_IN
        
        # update local Earth radii
        self.rn = rnCalc(x_hat[0])
        self.re = reCalc(x_hat[0])
        
        # set propogation time step
        self.dt = dt
        
    # given gyro measurement (wb_IB)_k-1, return (euler angles)_k as 1x3 array
    def updateEuler(self, wb_IB):
        self.wb_IB = np.array([self.x_hat[3], self.x_hat[4], self.x_hat[5]])
        self.wb_NB = self.wb_IB - self.wb_IN
        [r,p,y] = self.x_hat[6:9]
        A = np.array([
            [1, math.sin(r)*math.sin(p), math.cos(r)*math.sin(p)],
            [0, math.cos(r)*math.cos(p), -math.sin(r)*math.cos(p)],
            [0, math.sin(r), math.cos(r)]
            ])
        A = A/math.cos(p)
        rpy_new = self.x_hat[6:9] + self.dt*A @ self.wb_NB
        return rpy_new
    
    # given accelerometer measurement (f_B)_k, return v_N as 1x3 array
    def updateVelocity(self, f_B):
        f_N = self.DCM @ f_B
        g = gravityCalc(self.x_hat[0], self.x_hat[2])
        g_N = np.array([0,0,g])
        v_N = self.x_hat[3:6]
        v_N_dot = f_N + g_N - navpy.skew(2*self.wn_IE + self.wn_EN) @ v_N
        return v_N + self.dt*v_N_dot
    
    # using estimated velocity, propogate position
    def updatePosition(self, v_N):
        p_E_dot = np.diag([1/(self.rn + self.x_hat[2]), 
                           1/((self.re + self.x_hat[2])*math.cos(self.x_hat[0])),
                           -1]) @ v_N
        p_E = self.x_hat[0:3] + self.dt*p_E_dot
        return p_E
   
def wn_IECalc(x_hat):
    return 7.292115e-5*np.array([math.cos(x_hat[0]), 0, -math.sin(x_hat[0])])

def wn_ENCalc(x_hat):
    return np.array([x_hat[4]/(reCalc(x_hat[0])+x_hat[2]),
                  x_hat[3]/(rnCalc(x_hat[0])+x_hat[2]),
                  -x_hat[4]*math.tan(x_hat[0])/(reCalc(x_hat[0])+x_hat[2])
                  ])

# calculate DCM matrix given roll pitch yaw Euler angles
def Euler2DCM(rpy):
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]
    return np.array([[math.cos(p)*math.cos(y), math.sin(r)*math.sin(p)*math.cos(y)-math.cos(r)*math.sin(y), math.cos(r)*math.sin(p)*math.cos(y) + math.sin(r)*math.sin(y)],
                    [math.cos(p)*math.sin(y), math.sin(r)*math.sin(p)*math.sin(y) + math.cos(r)*math.cos(y), math.cos(r)*math.sin(p)*math.sin(y) - math.sin(r)*math.cos(y)],
                    [-math.sin(p), math.sin(r)*math.cos(p), math.cos(r)*math.cos(p)]])

# calculate meridian (North South) radius of curvature
def rnCalc(lat):
    a = 6378137
    f = 1/298.257223563
    e = math.sqrt(f*(2-f))
    return a*(1-math.pow(e,2))/math.pow( (1-math.pow(e,2)*math.pow(math.sin(lat),2)) , 1.5)

# calculate prime vertical (East West) radius of curvature
def reCalc(lat):
    a = 6378137
    f = 1/298.257223563
    e = math.sqrt(f*(2-f))
    return a/math.sqrt(1-math.pow(e,2)*math.pow(math.sin(lat), 2))

def gravityCalc(lat, alt):
    g0 = 9.7803253359*(1+0.001931853*math.pow(math.sin(lat),2)/
                       math.sqrt(0.00669438*math.pow(math.sin(lat),2)))
    return g0*(1-(3.157042870579883e-7 - 2.102689650440234e-9*math.pow(math.sin(lat),2))
               *alt + 7.374516772941995e-14*math.pow(alt,2))
             
def matprint(A):
    # Print the matrix without brackets or commas
    for row in A:
        for num in row:
            print(num, end=' ')
        print()  # Move to the next line after each row