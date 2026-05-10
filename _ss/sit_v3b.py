import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QGridLayout, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
import pyqtgraph as pg
from threading import Thread
import queue

class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Recorder")
        self.setGeometry(100, 100, 1200, 800)

        # Audio parameters
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.recording_buffer = []
        self.playback_buffer = []
        self.next_buffer = []
        self.is_recording = False
        self.is_playing = False
        self.recording_level = 0.7  # Default 70%
        self.playback_level = 0.7   # Default 70%
        self.is_first_recording = True
        self.iteration_count = 0
        
        # Display buffer for visualization
        self.display_buffer = np.zeros(self.sample_rate * 5)  # 5 seconds buffer
        self.buffer_position = 0
        
        # Audio devices
        self.input_device = None
        self.output_device = None
        self.stream = None
        
        # Analysis parameters
        self.audio_queue = queue.Queue()
        self.window_size = 4096  # FFT window size
        self.fft_data = np.zeros(self.window_size // 2 + 1)
        
        # Setup UI
        self.setup_ui()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizations)
        self.timer.start(50)  # 20 fps

    def normalize_audio(self, audio_data):
        """Normalize audio to peak at 1.0"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

    def save_iteration(self, audio_data, iteration_number):
        """Save the current iteration as a WAV file"""
        audio_int16 = np.int16(audio_data * 32767)
        filename = f"last_iteration{iteration_number}.wav"
        wavfile.write(filename, self.sample_rate, audio_int16)
        print(f"Saved iteration {iteration_number} to {filename}")

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

        # Main display area with controls and plots
        display_layout = QHBoxLayout()

        # Controls section (sliders and LEDs)
        controls_layout = QVBoxLayout()
        
        # Sliders with labels showing current values
        slider_layout = QGridLayout()
        
        # Recording level slider and label
        self.rec_value_label = QLabel("Rec: 70%")
        self.rec_slider = QSlider(Qt.Orientation.Vertical)
        self.rec_slider.setMinimum(0)
        self.rec_slider.setMaximum(100)
        self.rec_slider.setValue(70)  # Default 70%
        self.rec_slider.valueChanged.connect(self.update_recording_level)
        
        # Playback level slider and label
        self.play_value_label = QLabel("Play: 70%")
        self.play_slider = QSlider(Qt.Orientation.Vertical)
        self.play_slider.setMinimum(0)
        self.play_slider.setMaximum(100)
        self.play_slider.setValue(70)  # Default 70%
        self.play_slider.valueChanged.connect(self.update_playback_level)
        
        slider_layout.addWidget(self.rec_value_label, 0, 0)
        slider_layout.addWidget(self.play_value_label, 0, 1)
        slider_layout.addWidget(self.rec_slider, 1, 0)
        slider_layout.addWidget(self.play_slider, 1, 1)
        
        # Make sliders taller
        self.rec_slider.setMinimumHeight(200)
        self.play_slider.setMinimumHeight(200)
        
        # LED indicators below sliders
        led_layout = QHBoxLayout()
        self.record_led = LEDIndicator()
        self.playback_led = LEDIndicator()
        led_layout.addWidget(self.record_led)
        led_layout.addWidget(self.playback_led)
        
        controls_layout.addLayout(slider_layout)
        controls_layout.addLayout(led_layout)
        controls_layout.addStretch()  # This pushes everything to the top
        
        # Add controls to main display layout
        display_layout.addLayout(controls_layout)
        
        # Plots layout
        plots_layout = QHBoxLayout()
        
        # Waveform plot with fixed scaling
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setTitle("Waveform")
        self.waveform_plot.setLabel('left', "Amplitude")
        self.waveform_plot.setLabel('bottom', "Time (s)")
        self.waveform_plot.setYRange(-0.03, 0.03, padding=0)  # Fixed Y range
        self.waveform_plot.setXRange(0, 5, padding=0)        # Fixed 5-second window
        self.waveform_plot.disableAutoRange()  # Disable auto-scaling
        self.waveform_curve = self.waveform_plot.plot(pen='b')
        
        # FFT plot
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setTitle("Frequency Spectrum")
        self.fft_plot.setLabel('left', "Magnitude (dB)")
        self.fft_plot.setLabel('bottom', "Frequency (Hz)")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.setXRange(np.log10(20), np.log10(20000))  # Audible frequency range
        self.fft_plot.setYRange(-100, 0)    # Typical dB range
        self.fft_curve = self.fft_plot.plot(pen='r')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.5) #add grid

        plots_layout.addWidget(self.waveform_plot)
        plots_layout.addWidget(self.fft_plot)
        
        # Add plots to main display layout
        display_layout.addLayout(plots_layout)
        
        # Add everything to main layout
        main_layout.addLayout(display_layout)

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
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        if self.input_device is not None and self.output_device is not None:
            try:
                def audio_callback(indata, outdata, frames, time, status):
                    if status:
                        print(status)
                    
                    # Get current audio data
                    current_data = indata[:, 0]
                    
                    # Update display buffer
                    end_position = self.buffer_position + len(current_data)
                    if end_position <= len(self.display_buffer):
                        self.display_buffer[self.buffer_position:end_position] = current_data
                    else:
                        wrap_size = len(self.display_buffer) - self.buffer_position
                        self.display_buffer[self.buffer_position:] = current_data[:wrap_size]
                        self.display_buffer[:len(current_data)-wrap_size] = current_data[wrap_size:]
                    
                    self.buffer_position = (self.buffer_position + len(current_data)) % len(self.display_buffer)
                    
                    # Queue data for FFT
                    self.audio_queue.put(current_data)
                    
                    if self.is_recording and self.is_first_recording:
                        # Initial recording
                        scaled_data = current_data * self.recording_level
                        self.recording_buffer.extend(scaled_data)
                    
                    if self.is_playing:
                        if len(self.playback_buffer) >= frames:
                            # Play current iteration
                            outdata[:] = np.array(self.playback_buffer[:frames]).reshape(-1, 1) * self.playback_level
                            self.playback_buffer = self.playback_buffer[frames:]
                            
                            # Record next iteration
                            scaled_input = current_data * self.recording_level
                            self.next_buffer.extend(scaled_input)
                            
                            # Prepare next iteration
                            if len(self.playback_buffer) < frames:
                                # Normalize and save the just-completed iteration
                                if len(self.next_buffer) > 0:
                                    normalized_buffer = self.normalize_audio(np.array(self.next_buffer))
                                    self.save_iteration(normalized_buffer, self.iteration_count + 1)
                                    self.next_buffer = normalized_buffer.tolist()
                                
                                self.iteration_count += 1
                                self.iteration_label.setText(f"Iteration: {self.iteration_count}")
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
        if not self.audio_queue.empty():
            # Get latest audio data for FFT
            fft_data = []
            while not self.audio_queue.empty():
                fft_data.extend(self.audio_queue.get())
            
            # Update waveform using display buffer
            if self.buffer_position > 0:
                display_data = np.concatenate([
                    self.display_buffer[self.buffer_position:],
                    self.display_buffer[:self.buffer_position]
                ])
            else:
                display_data = self.display_buffer.copy()
            
            times = np.linspace(0, 5, len(self.display_buffer))
            self.waveform_curve.setData(times, display_data)
            
            # Update FFT
            if len(fft_data) >= self.window_size:
                # Get the latest chunk of data
                x = fft_data[-self.window_size:]
                
                # Apply window and compute FFT
                window = np.hanning(self.window_size)
                windowed_data = x * window
                spectrum = np.fft.rfft(windowed_data)
                frequencies = np.fft.rfftfreq(self.window_size, 1/self.sample_rate)
                
                # Compute power spectrum in dB
                magnitude = np.abs(spectrum)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
                
                # Apply some averaging/smoothing
                self.fft_data = 0.7 * self.fft_data + 0.3 * magnitude_db
                
                # Update FFT display
                self.fft_curve.setData(frequencies, self.fft_data)

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
                # Stop recording, normalize, save and start playback loop
                self.is_recording = False
                self.is_first_recording = False
                
                # Normalize and save initial recording
                normalized_buffer = self.normalize_audio(np.array(self.recording_buffer))
                self.save_iteration(normalized_buffer, 1)
                self.recording_buffer = normalized_buffer.tolist()
                
                # Start playback
                self.is_playing = True
                self.playback_buffer = self.recording_buffer.copy()
                self.next_buffer = []
                self.iteration_count = 1
                self.iteration_label.setText("Iteration: 1")
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        super().closeEvent(event)


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