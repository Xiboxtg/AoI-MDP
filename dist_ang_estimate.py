import numpy as np
import math
import random
import matplotlib.pyplot as plt


class dist_ang_estimate():
    def __init__(self, sound_speed=1500, Frequency=8000, max_distance=200, pulse_width=20,
                 doa_frequency=10000, doa_M=20):
        # Delay estimation related parameters.
        self.sound_speed = sound_speed
        self.Frequency = Frequency
        self.pulse_interval = math.ceil(max_distance*1.2*2 / sound_speed * Frequency/1000) * 1000
        self.pulse_width = pulse_width
        self.noise_stddev = 0.05
        # Generate Gaussian pulses.
        theta = pulse_width / (1.414 * 6.9077) # Variance of Gaussian pulse
        miu = self.pulse_width / 2  # Mean value of Gaussian pulse.
        pulse_nscal = np.array(range(self.pulse_width))
        gauss_pulse = np.exp(-(pulse_nscal - miu) ** 2 / 2 / (theta ** 2))
        self.guass_pulse = gauss_pulse
        # Parameters related to azimuth estimation.
        self.doa_frequency = doa_frequency
        self.doa_M = doa_M
        self.doa_d = sound_speed / self.doa_frequency / 2

    def dist_estimate(self, signal, pulse):

        N = len(signal)
        M = len(pulse)
        J = np.zeros(N-M+1)
        for n0 in range(N-M+1):
            signal_dat = signal[n0:n0+M]
            J[n0] = np.dot(signal_dat, pulse)
        # line1 = plt.plot(range(N-M+1), J)
        n0hat = np.argmax(J)
        time_delay = n0hat / self.Frequency / 2
        distance = time_delay * self.sound_speed
        return distance, time_delay

    def angle_estimate(self, signal_sametime, angle_num=1):

        angle_axis = np.linspace(start=0, stop=157, num=158) / 100
        Is = np.zeros(len(angle_axis))
        n = np.array(range(self.doa_M))
        for i in range(len(angle_axis)):
            ang_temp = 2 * np.pi * self.doa_frequency * self.doa_d / self.sound_speed * np.cos(angle_axis[i]) * n
            Is[i] = np.abs(np.dot(signal_sametime, np.exp(-1j * ang_temp))) ** 2 / self.doa_M

        local_max_index = []
        for i in range(len(angle_axis)):
            if angle_axis[i] > 0.02 and angle_axis[i] < 1.55:
                if Is[i] > Is[i-1] and Is[i] > Is[i+1]:
                    local_max_index.append(i)
        index_sort = np.argsort(Is[local_max_index])
        index = index_sort[-angle_num:]
        angle = []
        for i in range(angle_num):
            angle.append(angle_axis[local_max_index[index[i]]])
        return angle

    def dist_estimate_dd(self, distance_real):
        # Signal delay.
        n0 = distance_real * 2 / self.sound_speed * self.Frequency
        signal_delay = np.zeros(self.pulse_interval)
        signal_delay[int(n0):int(n0+self.pulse_width)] = self.guass_pulse
        # Generate noise.
        noise = np.random.normal(0, self.noise_stddev, size=(self.pulse_interval))
        # Generate signal.
        signal = signal_delay + noise
        # line2 = plt.plot(range(self.pulse_interval), signal)
        distance_estimate, time_delay = self.dist_estimate(signal, self.guass_pulse)
        # print(distance_real, distance_estimate)
        return distance_estimate, time_delay

    def angle_estimate_aa(self, angle):

        A = 1 + 0.2 * random.random()
        phi = np.pi/4 * (1 + random.random())

        noise = np.random.normal(0, self.noise_stddev, size=self.doa_M)
        signal_sametime = np.zeros(self.doa_M)
        for i in range(self.doa_M):
            s0 = A * np.cos(2 * np.pi * self.doa_frequency * self.doa_d / self.sound_speed *
                               np.cos(angle) * i + phi)
            signal_sametime[i] = s0 + noise[i]
        angle_est = self.angle_estimate(signal_sametime)
        # print(angle, angle_est)
        return angle_est

    def position_estimate(self, o_position):

        o_position = np.array(o_position)
        #auv_num = o_position.shape[0]
        e_position = np.zeros(2)
        o_dist = np.sqrt(o_position[0] ** 2 + o_position[1] ** 2)
        e_dist, time_delay = self.dist_estimate_dd(o_dist)
        o_ang = np.arctan2(o_position[1], o_position[0])
        e_ang = self.angle_estimate_aa(o_ang)
        e_position[0] = e_dist * np.cos(e_ang)
        e_position[1] = e_dist * np.sin(e_ang)
        return e_position, time_delay

# This is used to test the effectiveness of the algorithm.
if __name__ == "__main__":
    dist_ang_class = dist_ang_estimate()
    o_position = [[5, 120],[32,120]]
    e_position = dist_ang_class.position_estimate(o_position)
    print(o_position, e_position)