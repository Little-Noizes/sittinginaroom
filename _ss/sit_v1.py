import sys
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
import pyqtgraph as pg
from threading import Thread
import queue

class FrequencyAnalyzer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.window_size = 8192  # Large window for better low-frequency resolution
        self.hop_length = self.window_size // 4
        self.freq_bins = np.fft.rfftfreq(self.window_size, 1/sample_rate)
        self.peak_history = []  # Store peak frequencies over time
        
    def find_peaks_in_spectrum(self, spectrum, num_peaks=8):
        # Find peaks with minimum distance to avoid clustering
        peaks, properties = find_peaks(spectrum, 
                                     height=np.max(spectrum)/10,
                                     distance=20,
                                     prominence=np.max(spectrum)/20)
        
        peak_frequencies = self.freq_bins[peaks]
        peak_magnitudes = spectrum[peaks]
        
        # Sort by magnitude and get top peaks
        sorted_indices = np.argsort(peak_magnitudes)[-num_peaks:]
        return peak_frequencies[sorted_indices], peak_magnitudes[sorted_indices]

class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Room Mode Analyzer")
        self.setGeometry(100, 100, 1500, 900)

        # Audio parameters
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.recording_buffer = []
        self.playback_buffer = []
        self.is_recording = False
        self.is_playing = False
        self.recording_level = 1.0
        self.playback_level = 1.0
        
        # Analysis parameters
        self.freq_analyzer = FrequencyAnalyzer(self.sample_rate)
        self.audio_queue = queue.Queue()
        self.spectrogram_data = []
        
        # Setup UI
        self.setup_ui()
        
        # Setup audio
        self.setup_audio()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizations)
        self.timer.start(50)  # 20 FPS

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create visualization layout
        vis_layout = QHBoxLayout()
        
        # Left panel for waveform and peak frequencies
        left_panel = QVBoxLayout()
        
        # Waveform plot
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setTitle("Waveform")
        self.waveform_plot.setLabel('left', "Amplitude")
        self.waveform_plot.setLabel('bottom', "Time (s)")
        self.waveform_curve = self.waveform_plot.plot(pen='b')
        
        # Peak frequencies display
        self.peak_freq_widget = pg.PlotWidget()
        self.peak_freq_widget.setTitle("Detected Room Modes")
        self.peak_freq_widget.setLabel('left', "Magnitude (dB)")
        self.peak_freq_widget.setLabel('bottom', "Frequency (Hz)")
        self.peak_freq_widget.setLogMode(x=True, y=False)
        self.peak_freq_widget.setXRange(20, 2000)  # Focus on audible room modes
        self.peak_freq_scatter = self.peak_freq_widget.plot([], [], pen=None, symbol='o')
        
        left_panel.addWidget(self.waveform_plot)
        left_panel.addWidget(self.peak_freq_widget)
        
        # Spectrogram
        spec_layout = QVBoxLayout()
        self.spec_plot = pg.ImageView()
        self.spec_plot.ui.roiBtn.hide()
        self.spec_plot.ui.menuBtn.hide()
        self.spec_plot.setColorMap(pg.colormap.get('viridis'))
        spec_layout.addWidget(self.spec_plot)
        
        # Add layouts to main visualization layout
        vis_layout.addLayout(left_panel, stretch=1)
        vis_layout.addLayout(spec_layout, stretch=2)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Sliders
        slider_layout = QGridLayout()
        
        # Recording level slider
        self.rec_slider = QSlider(Qt.Orientation.Vertical)
        self.rec_slider.setMinimum(0)
        self.rec_slider.setMaximum(100)
        self.rec_slider.setValue(100)
        self.rec_slider.valueChanged.connect(self.update_recording_level)
        
        # Playback level slider
        self.play_slider = QSlider(Qt.Orientation.Vertical)
        self.play_slider.setMinimum(0)
        self.play_slider.setMaximum(100)
        self.play_slider.setValue(100)
        self.play_slider.valueChanged.connect(self.update_playback_level)
        
        slider_layout.addWidget(QLabel("Rec"), 0, 0)
        slider_layout.addWidget(QLabel("Play"), 0, 1)
        slider_layout.addWidget(self.rec_slider, 1, 0)
        slider_layout.addWidget(self.play_slider, 1, 1)
        
        # LED indicators
        led_layout = QHBoxLayout()
        self.record_led = LEDIndicator()
        self.playback_led = LEDIndicator()
        led_layout.addWidget(self.record_led)
        led_layout.addWidget(self.playback_led)
        
        controls_layout.addLayout(slider_layout)
        controls_layout.addLayout(led_layout)
        
        main_layout.addLayout(vis_layout)
        main_layout.addLayout(controls_layout)

    def setup_audio(self):
        def audio_callback(indata, outdata, frames, time, status):
            if self.is_recording:
                scaled_data = indata[:, 0] * self.recording_level
                self.recording_buffer.extend(scaled_data)
                self.audio_queue.put(scaled_data)
            
            if self.is_playing and len(self.playback_buffer) >= frames:
                outdata[:] = np.array(self.playback_buffer[:frames]).reshape(-1, 1) * self.playback_level
                self.playback_buffer = self.playback_buffer[frames:]
            else:
                outdata.fill(0)

        self.stream = sd.Stream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=audio_callback
        )
        self.stream.start()

    def update_visualizations(self):
        # Get latest audio data
        if not self.audio_queue.empty():
            data = []
            while not self.audio_queue.empty():
                data.extend(self.audio_queue.get())
            
            # Update waveform
            if len(data) > self.sample_rate * 5:  # 5 seconds window
                data = data[-self.sample_rate * 5:]
            times = np.linspace(0, len(data)/self.sample_rate, len(data))
            self.waveform_curve.setData(times, data)
            
            # Update spectrogram
            if len(data) > self.freq_analyzer.window_size:
                # Compute spectrogram with overlap
                frequencies, times, Sxx = signal.spectrogram(
                    data,
                    self.sample_rate,
                    nperseg=self.freq_analyzer.window_size,
                    noverlap=self.freq_analyzer.window_size * 3 // 4,
                    scaling='spectrum'
                )
                
                # Convert to dB scale
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                
                # Update spectrogram display with log frequency scale
                self.spec_plot.setImage(
                    Sxx_db,
                    scale=(times[-1]/Sxx_db.shape[1], np.log10(frequencies[-1]/frequencies[1])/Sxx_db.shape[0]),
                    pos=(0, np.log10(frequencies[1])),
                    autoRange=False
                )
                
                # Find and display peaks
                spectrum = np.mean(Sxx_db, axis=1)
                peak_freqs, peak_mags = self.freq_analyzer.find_peaks_in_spectrum(spectrum)
                self.peak_freq_scatter.setData(peak_freqs, peak_mags)

        # Update LEDs
        self.record_led.set_active(self.is_recording)
        self.playback_led.set_active(self.is_playing)

    def update_recording_level(self):
        self.recording_level = self.rec_slider.value() / 100.0

    def update_playback_level(self):
        self.playback_level = self.play_slider.value() / 100.0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if not self.is_recording:
                self.is_recording = True
                self.recording_buffer = []
            else:
                self.is_recording = False
                self.is_playing = True
                self.playback_buffer = self.recording_buffer.copy()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

class LEDIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(30, 30)
        self.active = False

    def set_active(self, active):
        self.active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.active:
            color = QColor(0, 255, 0)
        else:
            color = QColor(0, 100, 0)
            
        painter.setBrush(color)
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawEllipse(5, 5, 20, 20)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioVisualizer()
    window.show()
    sys.exit(app.exec())