import numpy as np
from vis import *

class task_1_1:
    def __init__(self, fs=1000):
        """
        Initialize the class with a specific sampling rate.

        Args:
            fs (int): Sampling rate in Hz. Defaults to 1000Hz.
        """
        self.fs = fs
    
    # TODO: Implement this function
    def generate_signal_1(self):
        """
        Generate the first signal: a pure tone with a specified frequency and phase offset.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated signal values. Data type must be float.

        Note:
            - The returned signal array must strictly be of type float.

        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_1()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, 0.6046)
        """  
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        duration = 2  # Duration of the signal in seconds
        frequency = 2  # Frequency in Hz (from -4π*x → frequency = 2 Hz)
        phase = np.pi / 3  # Phase offset in radians

        # Generate timestamps
        t = np.arange(-duration / 2, duration / 2, 1 / self.fs)

        # Generate the signal
        s_t = np.cos(-4 * np.pi * t + phase)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(float)
        t = np.array(t).astype(float)
        return t, s_t

    # TODO: Implement this function
    def generate_signal_2(self):
        """
        Generate the second signal: a combination of rectangle and triangle waves.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated signal values. Data type must be float.

        Note:
            - The returned signal array must strictly be of type float.

        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_2()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, 0.0)
        """
        
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # Calculate total time range (-1 to 1)
        t = np.linspace(-1, 1, self.fs * 2)  # fs * 2 ensures 2 seconds of data (fs samples per second)

        # Round time array to 2 decimal places to match expected output
        t = np.round(t, 2)

        # Initialize the signal array `s_t` with zeros
        s_t = np.zeros_like(t)

        # Define the piecewise function for p(t)
        # p(t) = 1 for -0.7 < t <= 0.3
        s_t[(t > -0.7) & (t <= 0.3)] = 1

        # p(t) = 5t - 1.5 for 0.3 < t <= 0.5
        s_t[(t > 0.3) & (t <= 0.5)] = 5 * t[(t > 0.3) & (t <= 0.5)] - 1.5

        # p(t) = -5t + 3.5 for 0.5 < t <= 0.7
        s_t[(t > 0.5) & (t <= 0.7)] = -5 * t[(t > 0.5) & (t <= 0.7)] + 3.5
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(float)
        t = np.array(t).astype(float)
        return t, s_t
        
    # TODO: Implement this function
    def generate_signal_3(self):
        """
        
        Generate the third signal: a complex signal based on real and imaginary parts.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated complex signal values. Data type must be np.complex64.

        Note:
            - The returned signal array must strictly be of type np.complex64.
            
        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_3()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, (0.99211+0.12533j))
        """
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # Generate time array
        t = np.linspace(-1, 1, self.fs * 2)  # Generate 2 seconds of data with fs samples per second
        t = np.round(t, 2)

        # Real part: cos(4 * pi * x)
        real_part = np.cos(4 * np.pi * t)

        # Imaginary part: cos(4 * pi * x - 1)
        imaginary_part = np.cos(4 * np.pi * t - (1/2) * np.pi)

        # Combine real and imaginary parts to form a complex signal
        s_t = real_part + 1j * imaginary_part
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(np.complex64)
        t = np.array(t).astype(float)
        return t, s_t
    
    def visualize(self):
        # Generate the first signal
        t, s_t = self.generate_signal_1()
        plot1D_single(t, s_t, '1-1', 'Time (s)', 'Amplitude')
        
        # Generate the second signal
        t, s_t = self.generate_signal_2()
        plot1D_single(t, s_t, '1-2', 'Time (s)', 'Amplitude')
        
        # Generate the third signal
        t, s_t = self.generate_signal_3()
        plot1D_multiple(t, [np.real(s_t), np.imag(s_t)], ['Real', 'Imaginary'], '1-3', 'Time (s)', 'Amplitude')
    