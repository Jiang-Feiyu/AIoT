import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class task_2_3:
    def __init__(self):
        """
        Initializes the task_2_3 class for performing FFT on different signal functions.
        """
        pass
    
    def func1(self, t):
        """
        Computes the value of the first signal function at given time points.
        """
        return np.cos(2 * np.pi * 50.99 * t) + np.cos(2 * np.pi * 51 * t) + np.sin(2 * np.pi * 51.02 * t)
    
    def func2(self, t):
        """
        Computes the value of the second signal function at given time points.
        """
        return np.cos(2 * np.pi * 51.2 * t) + np.sin(2 * np.pi * 1000.6 * t) + np.cos(2 * np.pi * 2000 * t)

    def get_freq_1(self):
        """
        Analyzes the first signal function using FFT to obtain its frequency spectrum.

        Before performing FFT, set the appropriate sampling rate `fs` and number of samples `N`.
        The time range for the signal is `[-N / (2 * fs), N / (2 * fs))`s.

        Returns:
            tuple: (fs, N, f) where 
            - `fs` (float64): the sampling rate
            - `N` (int): the number of samples,
            - `f` (np.float64): The frequency list (should be three frequencies).
        
        >>> test = task_2_3()
        >>> fs, N, f = test.get_freq_1()
        >>> fs != None, N != None
        (True, True)
        >>> f.sort()
        >>> [round(x, 2) for x in f]
        [50.99, 51.0, 51.02]
        """
        fs = None  # Define the appropriate sampling rate
        N = None  # Define the number of samples

        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Set the appropriate fs and N
        # Setting the appropriate sampling rate (fs) and number of samples (N)
        fs = 2000  # Sampling rate, must be greater than 2 * max frequency (Nyquist rate)
        N = 200000  # Number of samples, chosen to give sufficient frequency resolution
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        t = np.arange(-N / (2 * fs), N / (2 * fs),  1/fs)
        s_t = self.func1(t) # signal function
        
        f = np.zeros(3, dtype=np.float64) # frequency list
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Compute the FFT and get the frequency list
        # Compute the FFT and get the frequency list
        # Perform FFT
        fft_result = np.fft.fft(s_t)
        fft_freq = np.fft.fftfreq(N, d=1/fs)  # Generate corresponding frequencies

        # Only consider positive frequencies and their corresponding magnitudes
        positive_freqs = fft_freq[:N // 2]
        positive_magnitudes = np.abs(fft_result[:N // 2])

        # Find the top 3 frequencies with the highest magnitudes
        top_indices = np.argsort(positive_magnitudes)[-3:]  # Get indices of top 3 peaks
        f = positive_freqs[top_indices]  # Extract the corresponding frequencies
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        fs = float(fs)
        N = int(N)
        f = np.asarray(f).astype(np.float64)
        f = np.sort(f)
        
        return fs, N, f
    
    def get_freq_2(self):
        """
        Analyzes the second signal function using FFT to obtain its frequency spectrum.

        Before performing FFT, set the appropriate sampling rate `fs` and number of samples `N`.
        The time range for the signal is `[-N / (2 * fs), N / (2 * fs)]`s.

        Returns:
            tuple: (fs, N, f) where 
            - `fs` (float64): the sampling rate
            - `N` (int): the number of samples,
            - `f` (np.float64): The frequency list (should be three frequencies).
        
        >>> test = task_2_3()
        >>> fs, N, f = test.get_freq_2()
        >>> fs != None, N != None
        (True, True)
        >>> f.sort()
        >>> [round(x, 2) for x in f]
        [51.2, 1000.6, 2000.0]
        """
        fs = None  # Define the appropriate sampling rate
        N = None  # Define the number of samples

        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Set the appropriate fs and N
        # Setting the appropriate sampling rate (fs) and number of samples (N)
        fs = 5000  # Sampling rate, must be greater than 2 * max frequency (Nyquist rate)
        N = 50000  # Number of samples, chosen to give sufficient frequency resolution
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        t = np.arange(-N / (2 * fs), N / (2 * fs),  1/fs)
        s_t = self.func2(t) # signal function
        
        f = np.zeros(3, dtype=np.float64) # frequency list
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Compute the FFT and get the frequency list
        # Compute the FFT and get the frequency list
        # Perform FFT
        fft_result = np.fft.fft(s_t)
        fft_freq = np.fft.fftfreq(N, d=1/fs)  # Generate corresponding frequencies

        # Only consider positive frequencies and their corresponding magnitudes
        positive_freqs = fft_freq[:N // 2]
        positive_magnitudes = np.abs(fft_result[:N // 2])

        # Find the top 3 frequencies with the highest magnitudes
        top_indices = np.argsort(positive_magnitudes)[-3:]  # Get indices of top 3 peaks
        f = positive_freqs[top_indices]  # Extract the corresponding frequencies
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        fs = float(fs)
        N = int(N)
        f = np.asarray(f).astype(np.float64)
        f = np.sort(f)
        
        return fs, N, f
        
        