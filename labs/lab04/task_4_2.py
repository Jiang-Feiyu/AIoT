from scipy.signal import butter, sosfiltfilt
from scipy.signal import savgol_filter
from hampel import hampel
import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt


class task_4_2:
    def __init__(self, data_root="./data/"):
        """
        Initializes the task_4_1 class by loading signal data from a specified path.

        Parameters:
            data_root (str): The root directory where the signal data file is located.
                             The default value is "./data/".

        Attributes:
            data (np.ndarray): Loaded signal data.
            fs (int): Sampling rate in Hz, initialized from the loaded data.
        """
        file_1_n = "task_4_2_1.pickle"
        file_2_n = "task_4_2_2.pickle"
        with open(osp.join(data_root, file_1_n), "rb") as f:
            data1 = pickle.load(f)
        with open(osp.join(data_root, file_2_n), "rb") as f:
            data2 = pickle.load(f)
        self.data1, self.fs1 = data1["values"], data1["fs"]
        self.data2, self.fs2 = data2["values"], data2["fs"]
        self.clean1 = data1["clean"]
        self.clean2 = data2["clean"]

    def apply_filter_1(self):
        """
        Applies a specified filtering technique to smooth the noisy signal stored in self.data1.

        This method is designed to process signals sampled at self.fs1, specifically for the data
        provided in 'task_4_2_1.pickle' (self.data1). The goal is to filter out noise and extract meaningful
        information from the signal using an appropriate digital filter.

        The filter and its parameters should be chosen based on the characteristics of the noise
        and the signal of interest.

        Returns:
            np.ndarray: The filtered signal.
        
        >>> test = task_4_2(data_root="./data/")
        >>> filtered = test.apply_filter_1()
        >>> np.all(filtered != None)
        True
        >>> len(filtered) == len(test.clean1)
        True
        >>> rmse = test._compute_rmse(test.clean1, filtered)
        >>> rmse < 0.2
        True
        >>> snr = test._compute_snr(test.clean1, filtered)
        >>> snr > 20
        True
        >>> dv = test._compute_derivative_variation(filtered)
        >>> dv < 0.1
        True
        """
        filtered = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        from scipy import signal
        import numpy as np
        from hampel import hampel
    
        # 首先进行快速傅里叶变换分析信号频谱
        n = len(self.data1)
        fft_result = np.fft.rfft(self.data1)
        freqs = np.fft.rfftfreq(n, 1/self.fs1)
    
        # 应用理想低通滤波器（截断FFT）
        # 只保留0-2Hz之间的频率分量
        cutoff_freq = 2.0  # Hz
        fft_result[freqs > cutoff_freq] = 0
    
        # 反变换回时域
        initial_filtered = np.fft.irfft(fft_result, n)
    
        # 再应用hampel滤波器去除剩余的异常值
        hampel_result = hampel(initial_filtered, window_size=11, n_sigma=3.0)
        hampel_filtered = hampel_result.filtered_data
    
        # 应用Savitzky-Golay滤波器进行最终平滑
        window_length = 21  # 较大的窗口尺寸
        poly_order = 2  # 较低的多项式阶数，更平滑
        filtered = signal.savgol_filter(hampel_filtered, window_length, poly_order)
    
        # 最后应用一个双向巴特沃斯滤波器
        b, a = signal.butter(3, 0.05, 'low')
        filtered = signal.filtfilt(b, a, filtered)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        filtered = np.array(filtered, dtype=np.float64)
        return filtered
    
    def apply_filter_2(self):
        """
        Applies a different specified filtering technique to smooth the noisy signal stored in self.data2.

        Similar to apply_filter_1, this method targets signals sampled at self.fs2, but it is tailored
        for the data provided in 'task_4_2_2.pickle' (self.data2). It involves using a potentially different
        digital filter or parameters to address the unique characteristics of this signal and its
        associated noise.

        The chosen filter should effectively reduce noise while preserving the signal's integrity.

        Returns:
            np.ndarray: The filtered signal.
        
        >>> test = task_4_2(data_root="./data/")
        >>> filtered = test.apply_filter_2()
        >>> np.all(filtered != None)
        True
        >>> len(filtered) == len(test.clean2)
        True
        >>> rmse = test._compute_rmse(test.clean2, filtered)
        >>> rmse < 0.2
        True
        >>> snr = test._compute_snr(test.clean2, filtered)
        >>> snr > 35
        True
        >>> dv = test._compute_derivative_variation(filtered)
        >>> dv < 0.1
        True
        """
        filtered = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        from scipy import signal
        import numpy as np
        # 首先应用中值滤波器去除可能的脉冲噪声
        median_filtered = signal.medfilt(self.data2, kernel_size=5)
    
        # 然后使用Savitzky-Golay滤波器进行初步平滑
        window_length = 31  # 调整窗口长度
        polyorder = 3       # 多项式阶数
        savgol_filtered = signal.savgol_filter(median_filtered, window_length, polyorder)
    
        # 最后使用巴特沃斯低通滤波器进行最终平滑
        nyquist = 0.5 * self.fs2
        cutoff = 1.0 / nyquist  # 降低截止频率以进一步平滑
        order = 6               # 增加滤波器阶数以获得更陡峭的响应
    
        b, a = signal.butter(order, cutoff, btype='low')
        filtered = signal.filtfilt(b, a, savgol_filtered)

        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        filtered = np.array(filtered, dtype=np.float64)
        return filtered
    
    def _compute_snr(self, clean, filtered):
        """Helper function to compute SNR in dB."""
        signal_power = np.mean(clean ** 2)
        noise = filtered - clean
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    def _compute_rmse(self, clean, filtered):
        """Helper function to compute RMSE."""
        return np.sqrt(np.mean((filtered - clean) ** 2))

    def _compute_derivative_variation(self, signal):
        """Helper function to compute standard deviation of first derivative."""
        derivative = np.diff(signal)
        return np.std(derivative)