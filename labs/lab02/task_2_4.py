import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Radar parameters
c = 3e8           # Speed of light (m/s)
fc = 1e9          # Carrier frequency (Hz) 
B = 1.5e9         # Bandwidth (Hz)
T = 100e-6        # Chirp duration (s)
Fs = 2e6          # Sampling rate (Hz)
NUM_ANTENNAS = 4  # Number of antennas

class task_2_4:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_2_4 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            rx_fn (str): Filename for the received signal (task_2_4).
        """
        self.data_root = data_root
        self.rx_fn = "task_2_4.pickle"
        
        
        with open(osp.join(self.data_root, self.rx_fn), "rb") as f:
            self.rx_data = pickle.load(f)
        
        self.num_samples = self.rx_data.shape[1]
        
        
    def generate_transmitted_signal(self):
        r"""
        Generate the transmitted signal based on the received signal.
        
        The chirp signal is defined as:
        \[
            s(t) = \exp\left(j \cdot 2\pi \cdot (f_s \cdot + \dfrac{B}{2 \cdot T} \cdot t)\cdot t\right)
        \]
        
        Returns:
            tx (np.ndarray): The transmitted signal.
        
        >>> task = task_2_4()
        >>> tx = task.generate_transmitted_signal()
        >>> round(tx[-1].imag, 1)
        -0.7
        """
        tx = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Generate a time vector based on the sampling rate and chirp duration
        t = np.linspace(0, T, int(Fs * T), endpoint=False)

        # Compute the chirp signal using the given formula
        tx = np.exp(1j * 2 * np.pi * (fc * t + (B / (2 * T)) * t**2))
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return tx
    
    def compute_if_signal(self):
        r"""
        Compute the IF signal based on the received signal.
        
        if_signal is given by:
        \[
            if_signal = s(t) \cdot r^*(t)
        \]
        
        Returns:
            if_signal (np.ndarray): The IF signal.
        
        >>> task = task_2_4()
        >>> if_signal = task.compute_if_signal()
        >>> round(if_signal[-1][-1].imag, 1)
        -1.3
        """
        tx = self.generate_transmitted_signal()
        if_signal = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Access the received signal (self.rx_data)
        # self.rx_data has shape (NUM_ANTENNAS, num_samples)
        rx = self.rx_data  # Received signal
    
        # Compute the IF signal: if_signal = tx * conj(rx)
        # Ensure the transmitted signal is broadcasted to match the shape of the received signal
        if_signal = tx * np.conj(rx)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return if_signal
    
    def estimate_distance(self):
        """
        Estimate the distance based on the IF signal. In this case, there are two targets.
        
        Returns:
            distances (np.ndarray): The estimated distances (m) to the two targets in ascending order.
            range_fft (np.ndarray): The range FFT.
            range_bins (np.ndarray): The range bins corresponding to the range FFT (in meters).
        
        >>> task = task_2_4()
        >>> distances, _, _ = task.estimate_distance()
        >>> len(distances) == 2
        True
        """
        if_signal = self.compute_if_signal()
        distances = None
        range_fft = None # Range FFT
        range_bins = None # Range bins
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # 1. 对 IF 信号进行 FFT，并取幅值
        range_fft = np.fft.fft(if_signal, axis=-1)  # 对最后一个维度（时间轴）进行 FFT
        range_fft = np.abs(range_fft)  # 取幅值

        # 2. 对每个天线信号分别处理
        range_fft_avg = np.mean(range_fft, axis=0)  # 如果有多个天线，计算幅值的平均值（1-D）

        # 3. 生成频率轴（对应 range_bins）
        num_samples = self.num_samples
        T = 1e-6  # 假设信号的时长为 1 微秒（可根据实际任务调整）
        sampling_rate = num_samples / T  # 计算采样率
        freq_axis = np.fft.fftfreq(num_samples, d=1/sampling_rate)  # 频率轴
        c = 3e8  # 光速，单位：米/秒
        range_bins = freq_axis * c * T / 2  # 转换为距离轴

        # 4. 找到 FFT 的峰值（对应目标距离）
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(range_fft_avg, height=np.max(range_fft_avg) * 0.5)  # 阈值为 50% 的最大值
        distances = range_bins[peaks]  # 将峰值索引转换为距离
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<

        distances = np.sort(distances)
        return distances, range_fft, range_bins
    
    def estimate_AoA(self):
        """
        Estimate the angle of arrival based on the received signal.
        
        Returns:
            aoas (dict): A dictionary containing the estimated AoA for each target. You should keep one decimal place for the angles.
        
        >>> task = task_2_4()
        >>> aoas = task.estimate_AoA()
        >>> len(aoas) == 2, type(aoas) == dict
        (True, True)
        >>> all(isinstance(v, float) for v in aoas.values())
        True
        """
        _, range_fft, range_bins = self.estimate_distance()
        aoas = {}
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # 1. 获取目标的索引及距离
        from scipy.signal import find_peaks
        
        # 找到距离范围内的目标距离对应的范围单元索引
        distances, range_fft, range_bins = self.estimate_distance()
        target_indices = [np.abs(range_bins - d).argmin() for d in distances]

        # 2. 使用 2048 点 FFT 计算 AoA
        num_fft_points = 2048  # 2048 点 FFT
        for idx, distance in zip(target_indices, distances):
            # 提取目标距离对应的信号
            target_signal = range_fft[:, idx]  # 获取目标对应天线信号
            
            # 计算目标信号的相位差
            angle_fft = np.fft.fft(target_signal, num_fft_points)
            angle_idx = np.argmax(np.abs(angle_fft))  # 找到最大点对应的索引
            
            # 转换为 AoA (以度为单位)
            aoa = np.arcsin(angle_idx / num_fft_points) * 180 / np.pi
            aoas[round(distance, 1)] = round(aoa, 1)  # 保留一位小数
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        return aoas