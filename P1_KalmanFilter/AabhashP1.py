# -*- coding: utf-8 -*-
"""
Kalman Filter Design for Timing System
AEM667 - Navigation and Target Tracking 
Project-1


Created on Mon Sep 18 09:30:52 2023

@author: Aabhash Bhandari
"""


import numpy as np
from matplotlib import pyplot as plt


class KalmanFilter:
    """Main Class for Kalman Filters for this project"""

    def __init__(self, initial_x0: float, initial_x1: float, qk: float):
        """
        Contructor for Kalman Filter Class.

        Parameters
        ----------
        initial_x0 : float
            initial value for the first state
        initial_x1 : float
            initial value for the second state.
        qk : float
            parameter for the process covariance.

        Returns
        -------
        None.

        """

        self.x_est = np.array([[initial_x0], [initial_x1]])
        self._qk = qk

        self.y_est = 0
        self.P_est = np.eye(2)

        # results from the prediction step
        self._prior_P = None
        self._prior_x = None

        # saving the state and cov history
        self.x_history = []
        self.P_history = []

    def run_prediction_step(self, dt=0.01):
        """Run the prediction for the state and output using the state
        space model as well as the state covariance"""

        # predicting state
        F = np.array([[1, dt], [0, 1]])
        self._prior_x = F.dot(self.x_est)

        # updating the P matrix
        Q = np.array(
            [[0, 0], [0.5 * dt**2 * self._qk**2, dt * self._qk**2]]
        )
        self._prior_P = F.dot(self.P_est).dot(F.T) + Q

        # the measurement
        H = np.array([1, 0])
        self.y_est = H.dot(self._prior_x)

    def run_correction_step(self, meas_val, meas_var):
        """Run the correction step of Kalman filter: update the mean
        and covariance"""

        # innovation equation
        yk_bar = meas_val - self.y_est

        H = np.array([1, 0])
        R = meas_var
        S = H.dot(self._prior_P).dot((H.T).reshape(2, 1)) + R

        # Kalman Gain
        K = self._prior_P.dot((H.T).reshape(2, 1)) / S

        # new estimates of x and covariance
        self.x_est = self._prior_x + K * yk_bar
        self.P_est = (np.eye(2) - K.dot(H.reshape(1, 2))).dot(
            self._prior_P
        )

        # recording the estimates
        self.x_history.append(self.x_est)
        self.P_history.append(self.P_est)


class ClockSimulation:
    """Main class to simulate the state and measurement history"""

    def __init__(
        self,
        del_omega: np.array,
        del_theta: [np.array, float],
        timesteps: np.linspace,
        r: float,
    ):
        """


        Parameters
        ----------
        del_omega : np.array
            .
        del_theta : [np.array, float]
            .
        timesteps : np.linspace
            .
        r : float
            variance of the AWGN on measurements.

        Returns
        -------
        None.

        """
        self._del_omega = del_omega
        self._del_theta = del_theta
        self._r = r

        self.time = timesteps

        # dict to save stuff from KalmanFilter Class
        self.kf = {}

    def simulate_state_and_measurements(self):
        """Simulate the state and measurement history for given time"""

        # Equation 3: loop phase error
        self.del_phi = self._del_omega * self.time + self._del_theta

        # Equation 6: measurement model
        vk = np.random.normal(0, self._r, self.time.size)
        self.y_k = self.del_phi + vk

    def plot_simulated_state_and_measuremnts(self, extra_str=None):
        """Plot the simulated state and measurements."""

        # plotting the states
        plt.plot(self.time, self.del_phi)
        plt.title("Simulated State: {}".format(extra_str))
        plt.xlabel("time (s)")
        plt.ylabel("Loop Phase error")
        plt.show()

        # plotting the measurements
        plt.plot(self.time, self.y_k)
        plt.title("Simulated Measurements: {}".format(extra_str))
        plt.xlabel("time (s)")
        plt.ylabel("Loop Phase error")
        plt.show()

    def run_kalman_filter_estimation(self, name: str, qk: float):
        """
        Run Kalman Filter on the simulation

        Parameters
        ----------
        name : str
            dummy name to store stuff from KF.
        qk : float
            parameter for the process covariance matrix.

        Returns
        -------
        None.

        """

        # creating an instance of the KF
        kf = KalmanFilter(initial_x0=0, initial_x1=0, qk=qk)

        for index, t in enumerate(self.time):
            # run the prediction
            kf.run_prediction_step(dt=0.01)

            # run the correction
            kf.run_correction_step(self.y_k[index], self._r)

        # Save the attributes of the Kalman filter on this dict object
        # use the provided name as key
        self.kf[name] = kf

    def make_plots_for_submission(self, extra_str=None):
        """Make all the necessary plots required for the project"""

        # First figure showing the estimated state and truth
        figure1, axs = plt.subplots(2, figsize=(10, 5))
        axs[0].plot(time_array, self.del_phi, "k", label="True State")
        axs[1].plot(time_array, self._del_omega, "k", label="True State")

        # Looping through each Kalman filter
        for index, kf_name in enumerate(kf_names):
            # plotting estimate for first state
            axs[0].plot(
                time_array,
                [state[0] for state in self.kf[kf_name].x_history],
                colors[index],
                label="Estimate: {}".format(kf_name),
                linewidth=0.5,
            )
            axs[0].set_title(
                "True States and estimate: {}".format(extra_str)
            )
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Phase loop error")
            axs[0].legend(loc="upper right")

            # plotting estimate for second state
            axs[1].plot(
                time_array,
                [state[1] for state in self.kf[kf_name].x_history],
                colors[index],
                label="Estimate: {}".format(kf_name),
                linewidth=0.5,
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Frequency error")
            axs[1].legend(loc="upper right")
        plt.show()

        # Second figure showing the errors and uncertainties
        figure2, axs = plt.subplots(2, figsize=(10, 5))

        # Looping through each Kalman filter
        for index, kf_name in enumerate(kf_names):
            # Confidence intervals for state-0
            conf_intervals1 = 2 * np.sqrt(
                np.array([arr[0, 0] for arr in self.kf[kf_name].P_history])
            )
            # Confidence intervals for state-1
            conf_intervals2 = 2 * np.sqrt(
                np.array([arr[1, 1] for arr in self.kf[kf_name].P_history])
            )

            # Plotting the error for first state
            axs[0].plot(
                time_array,
                np.concatenate(
                    [state[0] for state in self.kf[kf_name].x_history]
                )
                - self.del_phi,
                colors[index],
                label="error: {}".format(kf_name),
                linewidth=0.5,
            )
            # Plotting the positive confidence interval
            axs[0].plot(
                time_array,
                conf_intervals1,
                colors[index],
                linestyle="--",
                label="95% conf: {}".format(kf_name),
                linewidth=0.3,
            )
            # Plotting the negative confidence interval
            axs[0].plot(
                time_array,
                -conf_intervals1,
                colors[index],
                linestyle="--",
                linewidth=0.3,
            )
            axs[0].set_title("Estimation Errors: {}".format(extra_str))
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Phase loop error")
            axs[0].legend(loc="upper right")

            # Plotting the error for second state
            axs[1].plot(
                time_array,
                np.concatenate(
                    [state[1] for state in self.kf[kf_name].x_history]
                )
                - self._del_omega,
                colors[index],
                label="error: {}".format(kf_name),
                linewidth=0.5,
            )
            # Plotting the positive confidence interval
            axs[1].plot(
                time_array,
                conf_intervals2,
                colors[index],
                linestyle="--",
                label="95% conf: {}".format(kf_name),
                linewidth=0.3,
            )
            # Plotting the negative confidence interval
            axs[1].plot(
                time_array,
                -conf_intervals2,
                colors[index],
                linestyle="--",
                linewidth=0.3,
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Frequency error")
            axs[1].legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    """Main Program for Project-1"""

    # All these 3 lists should be updated together.
    qk_vals = [1, 0.1, 0.01]
    colors = ["r", "g", "b"]
    kf_names = ["kf_with_qk={}".format(qk) for qk in qk_vals]

    time_array = np.linspace(start=0, stop=30, num=int(30 / 0.01))

    # =============================================================================
    #     # Creating simulation of state and measurement for Problem1
    # =============================================================================
    sim1 = ClockSimulation(
        del_omega=np.full(time_array.size, 0.25),
        del_theta=np.pi / 4,
        timesteps=time_array,
        r=0.05,
    )

    sim1.simulate_state_and_measurements()
    # sim1.plot_simulated_state_and_measuremnts("sim1")

    # Running 3 different Kalman Filters with different qk
    sim1.run_kalman_filter_estimation(kf_names[0], qk_vals[0])
    sim1.run_kalman_filter_estimation(kf_names[1], qk_vals[1])
    sim1.run_kalman_filter_estimation(kf_names[2], qk_vals[2])

    sim1.make_plots_for_submission("Sim-1")

    # =============================================================================
    #     # Creating simultation of state and measurement for Problem2
    # =============================================================================
    sim2 = ClockSimulation(
        del_omega=np.full(time_array.size, 0.25),
        del_theta=np.piecewise(
            time_array,
            [time_array < 15, ((time_array <= 30) & (time_array >= 15))],
            [np.pi / 4, -3 * np.pi / 3],
        ),
        timesteps=time_array,
        r=0.05,
    )

    sim2.simulate_state_and_measurements()

    # Running 3 different Kalman Filters with different qk
    sim2.run_kalman_filter_estimation(kf_names[0], qk_vals[0])
    sim2.run_kalman_filter_estimation(kf_names[1], qk_vals[1])
    sim2.run_kalman_filter_estimation(kf_names[2], qk_vals[2])

    sim2.make_plots_for_submission("Sim-2")
    # =============================================================================
    #
    # =============================================================================
    """
    qk is set based on how much we relatively trust the prediction part 
    of KF for estimating the state, which depends on how well we know the 
    dynamics/process.
    
    Since we are adding Q to FPF', value of qk adds more uncertainty
    in the state prediction. 
    

    """
