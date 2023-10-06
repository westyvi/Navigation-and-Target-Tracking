%% kalman filter function

% Class KalmanFilter implements a linear kalman filter process for one
% time step. It is defined upon init with a process noise covariance matrix
% R, a measurement noise covariance matrix Q, process matrix F, measurement
% matrix H. These are assumed time invariant for this implementation.

% KalmanFilter's method 'filter' takes in a posteriori state xhat and posteriori
% state covariance matrix P from the prior time step, and a measurement yk
% from the current time step. It does cool math and returns a new posteriori
% state xhat and posteriori covariance matrix P for the current time step.
classdef KalmanFilter667
    properties
        R
        Q
        F
        H
    end
    methods
        function obj = KalmanFilter667(R,Q,F,H)
            if nargin > 0
                obj.R = R;
                obj.Q = Q;
                obj.F = F;
                obj.H = H;
            else
                error("class KalmanFilter must be initialized with four matrices as input");
            end
        end
        function [xhat, Phat] = filter(obj,xhat_posteriori, P_posteriori, yk)
            % prediction step
            % x1(:,k) = F*x1(:,k-1);
            xhat_apriori = obj.F*xhat_posteriori;
            P_apriori = obj.F*P_posteriori*obj.F' + obj.Q;
        
            % calculate kalman gain
            residual = yk - obj.H*xhat_apriori; % innovation residual
            S = obj.H*P_apriori*obj.H' + obj.R; % innovation covariance
            K = S\P_apriori*obj.H';
        
            % correction step
            xhat_posteriori_k = xhat_apriori + K*residual;
            P_posteriori_k = (eye(2)-K*obj.H)*P_apriori;

            % send posterior results out with less long names
            xhat = xhat_posteriori_k;
            Phat = P_posteriori_k;
        end
    end
end