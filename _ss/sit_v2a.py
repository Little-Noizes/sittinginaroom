import sys
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QGridLayout, QComboBox)
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
        self.recording_buffer = []  # Current recording buffer
        self.playback_buffer = []   # Current playback buffer
        self.next_buffer = []       # Buffer for recording the next iteration
        self.is_recording = False
        self.is_playing = False
        self.recording_level = 1.0
        self.playback_level = 1.0
        self.is_first_recording = True  # Flag for initial recording
        
        # Audio devices
        self.input_device = None
        self.output_device = None
        self.stream = None
        
        # Analysis parameters
        self.freq_analyzer = FrequencyAnalyzer(self.sample_rate)
        self.audio_queue = queue.Queue()
        self.spec_buffer = []  # Buffer for spectrogram data
        self.iteration_count = 0  # Counter for recording/playback iterations
        
        # Setup UI
        self.setup_ui()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizations)
        self.timer.start(50)  # 20 FPS

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Device selection and iteration counter area
        top_layout = QGridLayout()
        
        # Input device selection
        top_layout.addWidget(QLabel("Input Device:"), 0, 0)
        self.input_device_combo = QComboBox()
        self.populate_device_list(self.input_device_combo, 'input')
        self.input_device_combo.currentIndexChanged.connect(self.input_device_changed)
        top_layout.addWidget(self.input_device_combo, 0, 1)
        
        # Output device selection
        top_layout.addWidget(QLabel("Output Device:"), 1, 0)
        self.output_device_combo = QComboBox()
        self.populate_device_list(self.output_device_combo, 'output')
        self.output_device_combo.currentIndexChanged.connect(self.output_device_changed)
        top_layout.addWidget(self.output_device_combo, 1, 1)
        
        # Iteration counter
        self.iteration_label = QLabel("Iteration: 0")
        self.iteration_label.setStyleSheet("QLabel { font-size: 16pt; }")
        top_layout.addWidget(self.iteration_label, 0, 2)
        
        main_layout.addLayout(top_layout)

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
        spec_widget = QWidget()
        spec_widget.setMinimumWidth(600)
        spec_layout.addWidget(spec_widget)
        
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setTitle("Spectrogram")
        self.spec_plot.setLabel('left', 'Frequency (Hz)')
        self.spec_plot.setLabel('bottom', 'Time (s)')
        self.spec_plot.setLogMode(y=True)
        self.spec_plot.setYRange(np.log10(20), np.log10(2000))
        spec_layout.addWidget(self.spec_plot)
        
        # Create and add the ImageItem for the spectrogram
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        
        # Set the color map directly
        colormap = pg.colormap.get('viridis')
        self.spec_img.setColorMap(colormap)
        
        # Add layouts to main visualization layout
        vis_layout.addLayout(left_panel, stretch=1)
        vis_layout.addLayout(spec_layout, stretch=2)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Sliders with labels showing current values
        slider_layout = QGridLayout()
        
        # Recording level slider and label
        self.rec_value_label = QLabel("Rec: 100%")
        self.rec_slider = QSlider(Qt.Orientation.Vertical)
        self.rec_slider.setMinimum(0)
        self.rec_slider.setMaximum(100)
        self.rec_slider.setValue(100)
        self.rec_slider.valueChanged.connect(self.update_recording_level)
        
        # Playback level slider and label
        self.play_value_label = QLabel("Play: 100%")
        self.play_slider = QSlider(Qt.Orientation.Vertical)
        self.play_slider.setMinimum(0)
        self.play_slider.setMaximum(100)
        self.play_slider.setValue(100)
        self.play_slider.valueChanged.connect(self.update_playback_level)
        
        slider_layout.addWidget(self.rec_value_label, 0, 0)
        slider_layout.addWidget(self.play_value_label, 0, 1)
        slider_layout.addWidget(self.rec_slider, 1, 0)
        slider_layout.addWidget(self.play_slider, 1, 1)
        
        # Make sliders taller
        self.rec_slider.setMinimumHeight(200)
        self.play_slider.setMinimumHeight(200)
        
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

    def populate_device_list(self, combo_box, device_type):
        devices = sd.query_devices()
        combo_box.clear()
        
        for i, device in enumerate(devices):
            if device_type == 'input' and device['max_input_channels'] > 0:
                combo_box.addItem(f"{device['name']} (in)", i)
            elif device_type == 'output' and device['max_output_channels'] > 0:
                combo_box.addItem(f"{device['name']} (out)", i)

    def input_device_changed(self, index):
        if index >= 0:
            self.input_device = self.input_device_combo.currentData()
            self.setup_audio()

    def output_device_changed(self, index):
        if index >= 0:
            self.output_device = self.output_device_combo.currentData()
            self.setup_audio()

    def setup_audio(self):
        # Stop existing stream if it exists
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        if self.input_device is not None and self.output_device is not None:
            try:
                def audio_callback(indata, outdata, frames, time, status):
                    if status:
                        print(status)
                    
                    if self.is_recording and self.is_first_recording:
                        # Initial recording
                        scaled_data = indata[:, 0] * self.recording_level
                        self.recording_buffer.extend(scaled_data)
                        self.audio_queue.put(scaled_data)
                    
                    if self.is_playing:
                        if len(self.playback_buffer) >= frames:
                            # Play current iteration
                            outdata[:] = np.array(self.playback_buffer[:frames]).reshape(-1, 1) * self.playback_level
                            self.playback_buffer = self.playback_buffer[frames:]
                            
                            # Record the microphone input (room response) for next iteration
                            scaled_input = indata[:, 0] * self.recording_level
                            self.next_buffer.extend(scaled_input)
                            self.audio_queue.put(scaled_input)
                            
                            # When current playback buffer is empty, prepare for next iteration
                            if len(self.playback_buffer) < frames:
                                self.iteration_count += 1
                                self.iteration_label.setText(f"Iteration: {self.iteration_count}")
                                
                                # Switch buffers for next iteration
                                self.playback_buffer = self.next_buffer.copy()
                                self.next_buffer = []
                        else:
                            outdata.fill(0)
                    else:
                        outdata.fill(0)

                self.stream = sd.Stream(
                    device=(self.input_device, self.output_device),
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=audio_callback
                )
                self.stream.start()
            except Exception as e:
                print(f"Error setting up audio: {e}")

    def update_visualizations(self):
        # Get latest audio data
        if not self.audio_queue.empty():
            data = []
            while not self.audio_queue.empty():
                data.extend(self.audio_queue.get())
            
            # Update waveform (5-second window)
            if len(data) > self.sample_rate * 5:
                data = data[-self.sample_rate * 5:]
            times = np.linspace(0, len(data)/self.sample_rate, len(data))
            self.waveform_curve.setData(times, data)
            
            # Update spectrogram (5-second window)
            if len(data) > self.freq_analyzer.window_size:
                # Compute spectrogram
                frequencies, times, Sxx = signal.spectrogram(
                    data,
                    self.sample_rate,
                    nperseg=2048,  # Smaller window for better time resolution
                    noverlap=1536,  # 75% overlap
                    scaling='spectrum',
                    mode='magnitude'
                )
                
                # Convert to dB scale with proper normalization
                Sxx_db = 20 * np.log10(np.maximum(Sxx, 1e-10))
                Sxx_db = np.maximum(Sxx_db, Sxx_db.max() - 60)  # 60 dB dynamic range
                
                # Update spectrogram image
                self.spec_img.setImage(
                    Sxx_db,
                    scale=(times[-1]/Sxx_db.shape[1], 
                          (np.log10(frequencies[-1]) - np.log10(frequencies[1]))/Sxx_db.shape[0]),
                    pos=(0, np.log10(frequencies[1]))
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
        self.rec_value_label.setText(f"Rec: {self.rec_slider.value()}%")

    def update_playback_level(self):
        self.playback_level = self.play_slider.value() / 100.0
        self.play_value_label.setText(f"Play: {self.play_slider.value()}%")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if not self.is_recording and not self.is_playing:
                # Start initial recording
                self.is_recording = True
                self.is_first_recording = True
                self.recording_buffer = []
                self.next_buffer = []
                self.iteration_count = 0
                self.iteration_label.setText("Iteration: 0")
            elif self.is_recording:
                # Stop recording and start playback loop
                self.is_recording = False
                self.is_first_recording = False
                self.is_playing = True
                self.playback_buffer = self.recording_buffer.copy()
                self.next_buffer = []
                self.iteration_count = 1
                self.iteration_label.setText("Iteration: 1")
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