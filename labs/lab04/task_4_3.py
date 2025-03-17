from scipy.signal import butter, sosfiltfilt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from hampel import hampel
import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt

class task_4_3:
    def __init__(self, data_root="./data/"):
        data_p = osp.join(data_root, "task_4_3.pickle")
        with open(data_p, 'rb') as f:
            data = pickle.load(f)
        self.am_signal = data['am_signal']
        self.imu_signal = data['imu_signal']
        self.fs = data['fs']
        self.fc = data['fc']
    
    def get_freq(self, s, fs):
        """
        Calculate the dominant frequency of the signal.

        Parameters:
            s (numpy.ndarray): 1D array.
            fs (float): Sampling frequency in Hz

        Returns:
            numpy.ndarray: 1D array of the dominant frequency in Hz. You should return the dominant two frequencies in ascending order. 
            If there is only one dominant frequency, you should return the same frequency twice.
        
        >>> task = task_4_3()
        >>> s = np.sin(2*np.pi*10*np.linspace(0, 1, 1000))
        >>> frequency = task.get_freq(s, 1000)
        >>> frequency != None
        array([ True,  True])
        >>> len(frequency) == 2
        True
        >>> np.subtract(frequency[-1], 10) < 1e-6
        True
        >>> s = np.sin(2*np.pi*10*np.linspace(0, 1, 1000)) + np.sin(2*np.pi*20*np.linspace(0, 1, 1000))
        >>> frequency = task.get_freq(s, 1000)
        >>> frequency != None
        array([ True,  True])
        >>> len(frequency) == 2
        True
        >>> np.subtract(frequency[-1], 20) < 1e-6
        True
        """
        frequency = []
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # 计算信号长度
        N = len(s)
    
        # 使用FFT计算信号的频谱
        # 我们只关心正频率，所以只取前半部分
        fft_vals = np.fft.rfft(s)
        fft_freqs = np.fft.rfftfreq(N, 1/fs)
    
        # 计算幅度谱
        magnitude = np.abs(fft_vals)
    
        # 由于DC分量（0频率）通常会很大，但我们更关心实际的信号频率，
        # 如果信号有DC分量，我们忽略它
        if len(magnitude) > 1:
            # 忽略0频率
            magnitude[0] = 0
    
        # 找出最大幅度对应的索引
        sorted_indices = np.argsort(magnitude)[::-1]  # 按幅度从大到小排序
    
        # 获取前两个主频率，如果只有一个，则重复
        if len(sorted_indices) >= 2:
            top_indices = sorted_indices[:2]
            frequency = fft_freqs[top_indices]
        else:
            # 如果频谱中只有一个点，则重复它
            frequency = np.array([fft_freqs[sorted_indices[0]], fft_freqs[sorted_indices[0]]])
    
        # 按升序排序
        frequency.sort()
    
        # 处理特殊情况：如果第一个和第二个频率幅度差异很小，它们可能是同一频率
        # 这里我们比较前两个最大值是否非常接近
        if len(sorted_indices) >= 2:
            max1 = magnitude[sorted_indices[0]]
            max2 = magnitude[sorted_indices[1]]
            # 如果第二个峰值小于第一个峰值的某个比例，认为只有一个主频率
            if max2 < max1 * 0.5:  # 阈值可以根据需要调整
                frequency = np.array([fft_freqs[sorted_indices[0]], fft_freqs[sorted_indices[0]]])
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        frequency = np.array(frequency, dtype=np.float64)
        return frequency
        
    def demodulate_signal(self, s, fs, fc):
        """
        Demodulate an amplitude-modulated (AM) signal to extract the message signal.

        Parameters:
            s (numpy.ndarray): 1D array of the input AM signal
            fs (float): Sampling frequency in Hz
            fc (float): Carrier frequency in Hz

        Input:
            The signal s should contain a low-frequency message (e.g., 10 Hz) modulated onto
            a higher-frequency carrier (fc), with possible noise. Length should match fs*t_duration.

        Returns:
            numpy.ndarray: 1D array of the demodulated and normalized message signal.
        
        >>> task = task_4_3()
        >>> message = task.demodulate_signal(task.am_signal, task.fs, task.fc)
        >>> len(message) == len(task.am_signal)
        True
        >>> np.mean(message) - 0 < 1e-6
        True
        >>> freq = task.get_freq(message, task.fs)
        >>> np.subtract(freq[-1], 10) < 1e-6
        True
        """
        demo_signal = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        from scipy import signal
        import numpy as np

        # 步骤1：将接收到的AM信号乘以载波信号以执行相干解调
        t = np.arange(len(s)) / fs  # 时间数组
        carrier = np.cos(2 * np.pi * fc * t)  # 生成与发送方相同的载波信号
    
        # 将接收信号与载波相乘
        multiplied_signal = s * carrier

        # 步骤2：使用低通滤波器提取消息信号
        # 设计低通滤波器，截止频率应该高于消息信号的最高频率但低于2*fc
        # 假设消息信号频率约为10Hz (根据测试用例)，设置截止频率为30Hz
        nyquist = 0.5 * fs
        cutoff = 30.0 / nyquist  # 归一化截止频率
    
        # 使用巴特沃斯低通滤波器
        b, a = signal.butter(6, cutoff, btype='low')
        demo_signal = signal.filtfilt(b, a, multiplied_signal)
    
        # 步骤3：去除DC分量并归一化
        # 去除直流分量
        demo_signal = demo_signal - np.mean(demo_signal)
    
        # 归一化信号
        if np.max(np.abs(demo_signal)) > 0:  # 避免除以零
            demo_signal = demo_signal / np.max(np.abs(demo_signal))
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        demo_signal = np.array(demo_signal, dtype=np.float64)
        return demo_signal
    

    
    def interpolate_signal(self, s):
        """
        Interpolate missing data points (NaN) in an IMU signal using linear interpolation.

        Parameters:
            s (numpy.ndarray): 1D array of the input signal

        Input:
            The signal s represents accelerometer data with gaps (NaNs)

        Returns:
            numpy.ndarray: 1D array of the interpolated signal, same length as input,
                           with NaN values replaced by linearly interpolated values based
                           on nearest valid neighbors.
                           
        >>> task = task_4_3()
        >>> interpolated_signal = task.interpolate_signal(task.imu_signal)
        >>> len(interpolated_signal) == len(task.imu_signal)
        True
        >>> np.isnan(interpolated_signal).sum() == 0
        True
        >>> np.subtract(interpolated_signal[1], 0.221) < 1e-3
        True
        >>> np.subtract(interpolated_signal[998], -0.311) < 1e-3
        True
        """
        interpolated_signal = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: 
        # 复制输入信号，避免修改原始数据
        interpolated_signal = s.copy()
    
        # 找出所有非NaN数据点的索引
        valid_indices = np.where(~np.isnan(s))[0]
    
        # 找出所有NaN数据点的索引
        nan_indices = np.where(np.isnan(s))[0]
    
        # 如果没有有效数据点或没有NaN数据点，则直接返回
        if len(valid_indices) == 0 or len(nan_indices) == 0:
            return interpolated_signal
    
        # 使用numpy.interp进行线性插值
        # 参数1：需要计算的x点（NaN的索引）
        # 参数2：已知的x点（非NaN的索引）
        # 参数3：已知的y值（非NaN的值）
        interpolated_values = np.interp(
            nan_indices,           # 需要插值的点的位置（NaN的索引）
            valid_indices,         # 已知数据点的位置（非NaN的索引）
            s[valid_indices]       # 已知数据点的值（非NaN的值）
        )
    
        # 用插值结果替换NaN值
        interpolated_signal[nan_indices] = interpolated_values
    
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        interpolated_signal = np.array(interpolated_signal, dtype=np.float64)
        return interpolated_signal
    
    def apply_filter(self, s, fs):
        """
        Smooth an IMU signal using a Butterworth low-pass filter to remove high-frequency noise.

        Parameters:
            s (numpy.ndarray): 1D array of the input signal.
            fs (float): Sampling frequency in Hz

        Input:
            The signal s should be a continuous IMU signal (e.g., after interpolation).

        Returns:
            numpy.ndarray: 1D array of the smoothed signal, same length as input.
        
        >>> task = task_4_3()
        >>> filtered_signal = task.apply_filter(task.imu_signal, task.fs)
        >>> len(filtered_signal) == len(task.imu_signal)
        True
        >>> freq = task.get_freq(filtered_signal, task.fs)
        >>> np.subtract(freq[-1], 1) < 1e-6
        True
        
        """
        interpolated_signal = self.interpolate_signal(self.imu_signal)
        filtered_signal = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        from scipy import signal
        import numpy as np
        
        # 设计巴特沃斯低通滤波器参数
        # 根据测试用例，我们需要保留1Hz的频率成分，设置截止频率为2Hz
        nyquist = 0.5 * fs
        cutoff = 2.0 / nyquist  # 归一化截止频率
        order = 4  # 滤波器阶数
    
        # 设计巴特沃斯低通滤波器
        b, a = signal.butter(order, cutoff, btype='low')
    
        # 应用双向滤波（零相位滤波）以避免相位失真
        filtered_signal = signal.filtfilt(b, a, interpolated_signal)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        filtered_signal = np.array(filtered_signal, dtype=np.float64)
        return filtered_signal
        
    