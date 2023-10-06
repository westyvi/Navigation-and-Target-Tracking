%% AEM 667 Project 1
% Joey Westermeyer 2023

% note: random measurement noise values vk are stored in a vector so they
% can be reused from problem 1 to 2. This is to keep the problem parameters
% the same except for the difference being examined- the jump in dtheta. 

% note: KF1 is kalman filter with qk = 0.01
%       KF2 is kalman filter with qk = 0.1
%       KF3 is kalman filter with qk = 1

clc
close all
clear all

% define plotting colors:
KF1c = 'c';
KF2c = 'm';
KF3c = 'k';
%% problem 1

% initialize constants
T = 0.01; % simulation time step, T = t/k
dw = 0.25;
dtheta = pi/4;
r = 0.05; % stddev of measurement noise
t0 = 0;
tfinal = 30;
xa = [dw*t0 + dtheta ;dw]; % initialize state with equation (3)
k = 1;
times(k) = t0;

% define discretized state equation matrices 
F = [1 T; 0 1]; % discretized state matrix
H = [1 0]; % discretized output matrix

% initialize first measurement
vk(k) = r*randn(1); % create measurement noise
ya(:,k) = H*xa(:,k) + vk(k); % sample

for t = t0+T:T:tfinal
    k = k+1;
    times(k) = t;
    % propogate real (simulated) state forward with no process noise
    xa(:,k) = F*xa(:,k-1);

    % sample random measurement noise
    vk(k) = r*randn(1);
    % sample state with
    ya(:,k) = H*xa(:,k) + vk(k);
end
%% problem 2
% same as problem one, but with discontinuity in dtheta at t = 15

% re-initialize variables not reused from problem 1
k = 1;
xb = [dw*t0 + dtheta ;dw]; % initialize state with equation (3)
yb(:,k) = H*xb(:,k) + vk(k); % sample

for t = t0+T:T:tfinal
    k = k+1;

    % propogate real (simulated) state forward with no process noise
    xb(:,k) = F*xb(:,k-1);
    if (t == 15)
        xb(1,k) = xb(1,k) - pi; % subtract pi from dtheta
    end

    % sample state with
    yb(:,k) = H*xb(:,k) + vk(k);
end
%% section 3
% design 3 kalman filters KF1, KF2, KF3 using equation 7 and 3 for process
% and measurement models respectively. Estimate state for both simulations

% note: filtering is normally done in real time with the simulation, but to
% keep filtering and simulation separate for this project, the filters will
% run at each time step on the simulation history data from parts a and b

% define constant model matrices and initial states for all kalman filters
R = r; % measurement noise matrix (scalar for this problem)
k = 1;
Phat_1a(:,:,k) = 1*eye(2); % initial state covariance (very uncertain) guess
Phat_2a = Phat_1a;
Phat_3a = Phat_2a;
Phat_1b = Phat_3a;
Phat_2b = Phat_1b;
Phat_3b = Phat_2b;
xhat_1a(:,k) = [0;0]; % initial state x
xhat_2a = xhat_1a;
xhat_3a = xhat_2a;
xhat_1b = xhat_3a;
xhat_2b = xhat_1b;
xhat_3b = xhat_2b;

% KF1: wk is white noise with constant qk = 0.01 for all k
qk = 0.01;
Q = [0 0; 0.5*T^2*qk^2 T*qk^2];
KF1 = KalmanFilter667(R,Q,F,H); % instantiate KF1

% KF2: wk is white noise with constant qk = 0.1 for all k
qk = 0.1;
Q = [0 0; 0.5*T^2*qk^2 T*qk^2];
KF2 = KalmanFilter667(R,Q,F,H); % instantiate KF2

% KF3: wk is white noise with constant qk = 1 for all k
qk = 1;
Q = [0 0; 0.5*T^2*qk^2 T*qk^2];
KF3 = KalmanFilter667(R,Q,F,H); % instantiate KF3

% run kalman filter at each time step, feeding in sensor data y(k) for each
% time step from simulation in parts (a) and (b)
for t = times(2:end)
    k = k+1;

    % simulation a data
    [xhat_1a(:,k), Phat_1a(:,:,k)] = KF1.filter(xhat_1a(:,k-1),Phat_1a(:,:,k-1),ya(k));
    [xhat_2a(:,k), Phat_2a(:,:,k)] = KF2.filter(xhat_2a(:,k-1),Phat_2a(:,:,k-1),ya(k));
    [xhat_3a(:,k), Phat_3a(:,:,k)] = KF3.filter(xhat_3a(:,k-1),Phat_3a(:,:,k-1),ya(k));

    % simulation b data
    [xhat_1b(:,k), Phat_1b(:,:,k)] = KF1.filter(xhat_1b(:,k-1),Phat_1b(:,:,k-1),yb(k));
    [xhat_2b(:,k), Phat_2b(:,:,k)] = KF2.filter(xhat_2b(:,k-1),Phat_2b(:,:,k-1),yb(k));
    [xhat_3b(:,k), Phat_3b(:,:,k)] = KF3.filter(xhat_3b(:,k-1),Phat_3b(:,:,k-1),yb(k));
end

%% section 4
% 4 figures with t = kT as x-axis, two subplots (one for each vector
% element):

%% (1) plot simulated state xk from part a and state estimates xhat for all three
% kalman filters
figure(1)
sgtitle('part (a) states')

subplot(2,1,1) % plot dphi
hold on
plot(times,xa(1,:),'k')
plot(times,xhat_1a(1,:),KF1c, times,xhat_2a(1,:),KF2c, times,xhat_3a(1,:),KF3c)
legend('simulated state', 'KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('loop phase error')

subplot(2,1,2) % plot domega
hold on
plot(times,xa(2,:),'k')
plot(times,xhat_1a(2,:),KF1c, times,xhat_2a(2,:),KF2c, times,xhat_3a(2,:),KF3c)
legend('simulated state', 'KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('frequency error')

%% (2) plot estimate error xk-xhat and 95% confidence intervals (diaganol
% values of +-2sqrt(Pk) from part a for all three filters
figure(2)
sgtitle('part (a) state estimate error and 95% confidence intervals')
subplot(2,1,1) % plot dphi
hold on
xlabel('time, s')
ylabel('loop phase error')

% plot confidence intervals (2*sqrt(element 1,1 of Phat))
 yconf_1a = [2*sqrt(squeeze(Phat_1a(1,1,:)))'+xhat_1a(1,:)-xa(1,:) flip(-2*sqrt(squeeze(Phat_1a(1,1,:)))'+xhat_1a(1,:)-xa(1,:))];
 yconf_2a = [2*sqrt(squeeze(Phat_2a(1,1,:)))'+xhat_2a(1,:)-xa(1,:) flip(-2*sqrt(squeeze(Phat_2a(1,1,:)))'+xhat_2a(1,:)-xa(1,:))];
 yconf_3a = [2*sqrt(squeeze(Phat_3a(1,1,:)))'+xhat_3a(1,:)-xa(1,:) flip(-2*sqrt(squeeze(Phat_3a(1,1,:)))'+xhat_3a(1,:)-xa(1,:))];
 p = fill([times flip(times)], yconf_1a,KF1c, [times flip(times)], yconf_2a, KF2c,[times flip(times)],yconf_3a, KF3c);
 p(1).FaceAlpha = .5;
 p(2).FaceAlpha = .2;
 p(3).FaceAlpha = .1;

% plot errors
plot(times,xhat_1a(1,:)-xa(1,:),KF1c, times,xhat_2a(1,:)-xa(1,:),KF2c, times,xhat_3a(1,:)-xa(1,:),KF3c)
legend('KF1', 'KF2', 'KF3')
ylim([-0.25 0.25])

subplot(2,1,2) % plot domega
hold on
% plot confidence intervals
 yconf_1a = [2*sqrt(squeeze(Phat_1a(2,2,:)))'+xhat_1a(2,:)-xa(2,:) flip(-2*sqrt(squeeze(Phat_1a(2,2,:)))'+xhat_1a(2,:)-xa(2,:))];
 yconf_2a = [2*sqrt(squeeze(Phat_2a(2,2,:)))'+xhat_2a(2,:)-xa(2,:) flip(-2*sqrt(squeeze(Phat_2a(2,2,:)))'+xhat_2a(2,:)-xa(2,:))];
 yconf_3a = [2*sqrt(squeeze(Phat_3a(2,2,:)))'+xhat_3a(2,:)-xa(2,:) flip(-2*sqrt(squeeze(Phat_3a(2,2,:)))'+xhat_3a(2,:)-xa(2,:))];
 p = fill([times flip(times)], yconf_1a,KF1c, [times flip(times)], yconf_2a, KF2c,[times flip(times)],yconf_3a, KF3c);
 p(1).FaceAlpha = .5;
 p(2).FaceAlpha = .2;
 p(3).FaceAlpha = .1;

% plot errors
plot(times,xhat_1a(2,:)-xa(2,:),KF1c, times,xhat_2a(2,:)-xa(2,:),KF2c, times,xhat_3a(2,:)-xa(2,:),KF3c)
legend('KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('frequency error')
ylim([-2 2])

%% (3) plot simulated state xk from part b and state estimates xhat for all three
% kalman filters
figure(3)
sgtitle('part (b) states')
subplot(2,1,1) % plot dphi
hold on
xlabel('time, s')
ylabel('loop phase error')
plot(times,xb(1,:),'k')
plot(times,xhat_1b(1,:),KF1c, times,xhat_2b(1,:),KF2c, times,xhat_3b(1,:),KF3c)
legend('state', 'KF1', 'KF2', 'KF3')

subplot(2,1,2) % plot domega
hold on
plot(times,xb(2,:),'k')
plot(times,xhat_1b(2,:),KF1c, times,xhat_2b(2,:), KF2c, times,xhat_3b(2,:),KF3c)
legend('state', 'KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('frequency error')

%% (4) plot estimate error xk-xhat and 95% confidence intervals (diaganol
% values of +-2sqrt(Pk) from part b for all three filters
figure(4)
sgtitle('part (b) state estimate error and 95% confidence intervals')
subplot(2,1,1) % plot dphi
hold on
% plot confidence intervals
% plot confidence intervals (2*sqrt(element 1,1 of Phat))
 yconf_1b = [2*sqrt(squeeze(Phat_1b(1,1,:)))'+xhat_1b(1,:)-xb(1,:) flip(-2*sqrt(squeeze(Phat_1b(1,1,:)))'+xhat_1b(1,:)-xb(1,:))];
 yconf_2b = [2*sqrt(squeeze(Phat_2b(1,1,:)))'+xhat_2b(1,:)-xb(1,:) flip(-2*sqrt(squeeze(Phat_2b(1,1,:)))'+xhat_2b(1,:)-xb(1,:))];
 yconf_3b = [2*sqrt(squeeze(Phat_3b(1,1,:)))'+xhat_3b(1,:)-xb(1,:) flip(-2*sqrt(squeeze(Phat_3b(1,1,:)))'+xhat_3b(1,:)-xb(1,:))];
 p = fill([times flip(times)], yconf_1b,KF1c, [times flip(times)], yconf_2b, KF2c,[times flip(times)],yconf_3b, KF3c);
 p(1).FaceAlpha = .5;
 p(2).FaceAlpha = .2;
 p(3).FaceAlpha = .1;

% plot errors
plot(times,xhat_1b(1,:)-xb(1,:),KF1c, times,xhat_2b(1,:)-xb(1,:),KF2c, times,xhat_3b(1,:)-xb(1,:), KF3c)
legend('KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('loop phase error')
ylim([-1 3])

subplot(2,1,2) % plot domega
hold on
% plot confidence intervals
 yconf_1b = [2*sqrt(squeeze(Phat_1b(2,2,:)))'+xhat_1b(2,:)-xb(2,:) flip(-2*sqrt(squeeze(Phat_1b(2,2,:)))'+xhat_1b(2,:)-xb(2,:))];
 yconf_2b = [2*sqrt(squeeze(Phat_2b(2,2,:)))'+xhat_2b(2,:)-xb(2,:) flip(-2*sqrt(squeeze(Phat_2b(2,2,:)))'+xhat_2b(2,:)-xb(2,:))];
 yconf_3b = [2*sqrt(squeeze(Phat_3b(2,2,:)))'+xhat_3b(2,:)-xb(2,:) flip(-2*sqrt(squeeze(Phat_3b(2,2,:)))'+xhat_3b(2,:)-xb(2,:))];
 p = fill([times flip(times)], yconf_1b,KF1c, [times flip(times)], yconf_2b, KF2c,[times flip(times)],yconf_3b, KF3c);
 p(1).FaceAlpha = .5;
 p(2).FaceAlpha = .2;
 p(3).FaceAlpha = .1;

 % plot errors
plot(times,xhat_1b(2,:)-xb(2,:), KF1c, times,xhat_2b(2,:)-xb(2,:), KF2c, times,xhat_3b(2,:)-xb(2,:), KF3c)
legend('KF1', 'KF2', 'KF3')
xlabel('time, s')
ylabel('frequency error')

%% part 5
% comment on and compare behavior, include opinion on what you would set qk
% as. 

% As can be seen in the plots, the kalman filter with qk = 1 (KF3) has the
% most noise in the 'steady state' portion of filtering. This makes sense,
% as KF3's Q (process noise) matrix has the highest values of the three
% filters, meaning the apriori covariance matrix has larger
% uncertainty values and thus the propogated state has higher uncertainty. 
% Since the propogated state has the highest uncertainty in KF3, this
% filter trusts the noisy sensor measurements the most, resulting in the
% noisiest output of the three filters by far when the filter's model
% matches the real process well, as in the part (a) simulations. However,
% this distrust of the modeled process in KF3 results in much faster
% adjustments to the estimated state when disturbances to the states occur
% that are not captured by the process model, as in part (b). This is
% because the filter, as stated before, has lower uncertainty on the sensor
% than its internal state prediction, so when the state unexpectedly jumps,
% KF3 weights the (very different from the last step) sensor measurement
% more than the other filters, and thus re-captures the changed state
% fastest. 

% It follows that KF2 has less noise in the steady state (part a) portions
% than KF3, and KF1 has the lowest noise in the output, as well as the 
% lowest uncertainty in the state, for part a. However, the second point of
% the prior paragraph also follows; KF2 recovers from the discontinous
% state jump slower than KF3, and KF1 takes the longest. This is because
% these filters have lower uncertainty values in their internal process 
% modeling, so they weight their propogated states more in the sensor
% fusion. Because of this, when the state experiences an unmodeled
% disturbance, these filters take longer to output the actual state, which
% is being successfully measured with noise. 

% For the parameters of this project, I would recommend selecting KF2 with
% qk = 0.1. This is because KF3 has such high process uncertainty it only
% has a slight advantage over taking the raw sensor data with no filtering,
% and KF1 is much slower to respond to unmodeled state disturbances.
% However, the specific desires of our clock network could change this
% option. If unmodeled state jumps such as in part (b) are not expected in
% this system, KF1 may be a better choice as it has even higher certainty
% in its output and lower output noise than KF2. And the contrary follows;
% if higher noise is acceptable in the output and the model is known to not
% capture many large discontinuities in states, KF3 may be a better choice.
% However, for general applications with parameters similar to this
% simulation I would recommend KF2 as it has
% significantly improved noise filtering of the states over KF3 and
% recovers from unmodeled disturbances much faster than KF1. 