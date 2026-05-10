import numpy as np
from scipy import signal

class FrequencyAnalyzer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.window_size = 2048  # Reduced for better real-time performance
        self.hop_length = self.window_size // 4
        self.freq_bins = np.fft.rfftfreq(self.window_size, 1/sample_rate)
        self.peak_history = []  # Store peak frequencies over time
        
    def find_peaks_in_spectrum(self, spectrum, num_peaks=8):
        # Find peaks with minimum distance to avoid clustering
        peaks, properties = signal.find_peaks(spectrum, 
                                     height=np.max(spectrum)/10,
                                     distance=20,
                                     prominence=np.max(spectrum)/20)
        
        peak_frequencies = self.freq_bins[peaks]
        peak_magnitudes = spectrum[peaks]
        
        # Sort by magnitude and get top peaks
        sorted_indices = np.argsort(peak_magnitudes)[-num_peaks:]
        return peak_frequencies[sorted_indices], peak_magnitudes[sorted_indices]