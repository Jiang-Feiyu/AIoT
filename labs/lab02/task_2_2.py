import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class task_2_2:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_2_2 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            spt_fn (str): Filename for the sum of pure tone signals (task_2_2_1).
            chirp_fn (str): Filename for the chirp signal (task_2_2_2).
            ecg_fn (str): Filename for the ECG signal (task_2_2_3).
            spt_data (dict): Loaded data for the sum of pure tone signals.
            chirp_data (dict): Loaded data for the chirp signal.
            ecg_data (dict): Loaded data for the ECG signal.
        """
        self.data_root = data_root
        self.spt_fn = "task_2_2_1.pickle"
        self.chirp_fn = "task_2_2_2.pickle"
        self.ecg_fn = "task_2_2_3.pickle"
        
        with open(osp.join(self.data_root, self.spt_fn), "rb") as f:
            self.spt_data = pickle.load(f)
        with open(osp.join(self.data_root, self.chirp_fn), "rb") as f:
            self.chirp_data = pickle.load(f)
        with open(osp.join(self.data_root, self.ecg_fn), "rb") as f:
            self.ecg_data = pickle.load(f)
        
    def get_freq_spt(self):
        """
        Analyze the sum of pure tone signals to determine the primary frequency components.

        Returns:
            freq (np.float64): An array of the three primary frequency components in descending order.
        
        >>> test = task_2_2()
        >>> f = test.get_freq_spt()
        >>> len(f) == 3
        True
        """
        s_t = self.spt_data["values"] # signal values
        fs = self.spt_data["fs"] # sampling frequency
        
        freq = np.zeros(3, dtype=np.float64) # (3,)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Perform FFT on the signal
        fft_result = np.fft.fft(s_t)
        fft_magnitude = np.abs(fft_result)  # Get the magnitude of the FFT
        
        # Compute the frequency axis
        N = len(s_t)  # Length of the signal
        freq_axis = np.fft.fftfreq(N, d=1/fs)
        
        # Extract positive frequency components
        positive_freq_idx = freq_axis >= 0
        fft_magnitude = fft_magnitude[positive_freq_idx]
        freq_axis = freq_axis[positive_freq_idx]
        
        # Find the indices of the top 3 frequencies
        top_indices = np.argsort(fft_magnitude)[-3:]  # Get indices of 3 highest magnitudes
        freq = freq_axis[top_indices]  # Get the corresponding frequencies
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        freq = np.sort(freq)[::-1]
        freq = np.squeeze(freq[:3]).astype(np.float64)
        return freq
    
    def get_bw_chirp(self):
        """
        Compute the bandwidth of the chirp signal.

        Returns:
            bw (float64): The bandwidth of the chirp signal in Hz. Format: float64.
            
        >>> test = task_2_2()
        >>> bw = test.get_bw_chirp()
        >>> (bw >= 100) & (bw <= 1000)
        True
        """
        s_t = self.chirp_data["values"] # signal values
        fs = self.chirp_data["fs"] # sampling frequency
        
        bw = 0
        #  chirp signal. The sampling rate is 16000 Hz. In this task, your task is to return the bandwidth of the chirp signal.
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Perform FFT to compute frequency spectrum
        fft_result = np.fft.fft(s_t)
        fft_magnitude = np.abs(fft_result)  # Get magnitude of FFT
        freqs = np.fft.fftfreq(len(s_t), 1 / fs)  # Compute frequency bins

        # Consider only the positive half of the spectrum
        positive_freqs = freqs[freqs >= 0]
        positive_magnitude = fft_magnitude[freqs >= 0]

        # Normalize the magnitude spectrum
        normalized_magnitude = positive_magnitude / np.max(positive_magnitude)

        # Define a threshold to determine the significant frequency range
        threshold = 0.01  # 1% of the max magnitude

        # Find indices where the magnitude exceeds the threshold
        significant_indices = np.where(normalized_magnitude >= threshold)[0]

        # Compute bandwidth as the frequency range of significant components
        if len(significant_indices) > 0:
            f_min = positive_freqs[significant_indices[0]]  # Minimum significant frequency
            f_max = positive_freqs[significant_indices[-1]]  # Maximum significant frequency
            bw = f_max - f_min  # Bandwidth
        else:
            bw = 0  # If no significant frequencies, bandwidth is 0
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        return bw 
    
    def get_heart_rate(self):
        """
        Determine the heart rate from the ECG signal.

        Returns:
            hr (float64): The heart rate in beats per minute (BPM).
        
        >>> test = task_2_2()
        >>> hr = test.get_heart_rate()
        >>> (hr >= 60) & (hr <= 90)
        True
        """
        s_t = self.ecg_data["values"] # signal values
        fs = self.ecg_data["fs"] # sampling frequency
        
        hr = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Detect peaks in the ECG signal
        # Typical heart rate range in Hz: 60 BPM to 90 BPM (1 Hz to 1.5 Hz)
        min_hr_hz = 60 / 60  # 1 Hz
        max_hr_hz = 90 / 60  # 1.5 Hz

        # Minimum and maximum distance between peaks (in samples)
        min_distance = int(fs / max_hr_hz)  # Minimum distance between peaks
        max_distance = int(fs / min_hr_hz)  # Maximum distance between peaks

        # Find peaks in the ECG signal
        peaks, _ = find_peaks(s_t, distance=min_distance)

        # Calculate the intervals between consecutive peaks (in seconds)
        peak_intervals = np.diff(peaks) / fs  # Time differences between peaks

        # Compute the average heart rate (in Hz)
        avg_hr_hz = 1 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0

        # Convert to BPM (Beats Per Minute)
        hr = avg_hr_hz * 60  # Convert Hz to BPM

        # Ensure the HR is a float64
        hr = np.float64(hr)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # A clip of ECG (electrocardiogram) signal. The sampling rate is 100 Hz.
        # Your task is to uncover the heart rate from ECG. Heartbeat is a periodic event, and the heart rate is the frequency that the heart beats. The heart rate of this participant is between 60 - 90 BPM
        # (Beat Per Second). You should return the heart rate in BPM.

        # you are recommended to only focus on those peaks whose corresponding fre- quencies lie in the reasonable range of your task, i.e. heart rate range.

        # Make sure hr is a float64
        if isinstance(hr, np.ndarray):
            if hr.size > 1:
                hr = hr[0]
            hr = hr.item()
        if isinstance(hr, list):
            if len(hr) > 1:
                hr = hr[0]
        hr = float(hr)
        return hr

if __name__ == "__main__":
    data_root = "./data/" # Change this to the directory where you store the data
    test = task_2_2(data_root=data_root)
    # ...