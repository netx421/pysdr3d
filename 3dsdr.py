import sys
from PyQt5.QtGui import QPalette, QColor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import Qt, QTimer, QThread
from rtlsdr import RtlSdr

class CaptureThread(QThread):
    def __init__(self, parent, sdr, fft_size):
        super().__init__(parent)
        self.parent = parent
        self.sdr = sdr
        self.fft_size = fft_size

    def run(self):
        while self.parent.running:
            samples = self.sdr.read_samples(self.fft_size)
            spectrum = np.fft.fft(samples)
            power_spectrum = 10 * np.log10(np.abs(spectrum) ** 2)

            # Append the new power spectrum to the beginning of the waterfall data
            self.parent.waterfall_data.insert(0, power_spectrum)

            # Truncate the data if it exceeds the maximum number of snapshots
            if len(self.parent.waterfall_data) > self.parent.max_snapshots:
                self.parent.waterfall_data.pop()

            self.parent.update_3d_waterfall()
            self.parent.update_2d_waterfall()

class WaterfallApp(QMainWindow):
    def __init__(self):
        super().__init__()
        app.setStyle('Fusion')
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Dark, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.Shadow, QColor(20, 20, 20))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))
        app.setPalette(dark_palette)
        # Set dark theme for Matplotlib plots
        plt.style.use('dark_background')

        # RTL-SDR configuration
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.048e6
        self.sdr.center_freq = 103e6
        self.sdr.gain = 'auto'

        # Parameters
        self.duration = 30
        self.fft_size = 1024
        self.max_snapshots = 30
        self.snapshot_duration = 0.1
        self.waterfall_data = []

        # Create custom colormaps
        self.standard_cmap = LinearSegmentedColormap.from_list(
            'vintage_amber_spectrum',
            [(0.4, 0.3, 0.05), (0.05, 0.05, 0.05), (1.0, 0.8, 0.2)],  # Black to Bright Amber
            N=256
        )


        self.high_def_cmap = LinearSegmentedColormap.from_list(
            'custom_reds_high_def',
            [(0.05, 0.02, 0.02), (0.5, 0.2, 0.1), (1.0, 0.8, 0.6)],  # White to Black
            N=256
        )

        self.current_cmap = self.standard_cmap  # Default to standard colormap

        # Create widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        self.canvas_3d = FigureCanvas(plt.figure(figsize=(8, 6)))  # You can adjust the figsize as needed(plt.figure())
        self.layout.addWidget(self.canvas_3d, 3)

        self.canvas_2d = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas_2d, 1)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_capture)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_capture)
        self.layout.addWidget(self.stop_button)

        self.toggle_cmap_button = QPushButton("Dark Matter")
        self.toggle_cmap_button.clicked.connect(self.toggle_colormap)
        self.layout.addWidget(self.toggle_cmap_button)

        self.frequency_label = QLabel("Enter Frequency (MHz):")
        self.layout.addWidget(self.frequency_label)

        self.frequency_entry = QLineEdit()
        self.layout.addWidget(self.frequency_entry)

        self.set_frequency_button = QPushButton("Set Frequency")
        self.set_frequency_button.clicked.connect(self.update_frequency)
        self.layout.addWidget(self.set_frequency_button)

        self.current_frequency_label = QLabel(f"Current Frequency: {self.sdr.center_freq / 1e6} MHz")
        self.layout.addWidget(self.current_frequency_label)

        self.central_widget.setLayout(self.layout)

        # Variables for controlling the application
        self.running = False
        self.capture_thread = None

        # Create QTimer for updating plots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_data)

    def toggle_colormap(self):
        # Toggle between standard and high-def colormaps
        if self.current_cmap == self.standard_cmap:
            self.current_cmap = self.high_def_cmap
        else:
            self.current_cmap = self.standard_cmap

    def update_frequency(self):
        new_frequency_str = self.frequency_entry.text()
        try:
            new_frequency = float(new_frequency_str) * 1e6
            self.sdr.center_freq = new_frequency
            self.frequency_entry.clear()
        except ValueError:
            pass

    def start_capture(self):
        if not self.running:
            self.running = True
            self.waterfall_data = []  # Clear existing data
            self.capture_thread = CaptureThread(self, self.sdr, self.fft_size)
            self.capture_thread.start()
            self.timer.start(int(self.snapshot_duration * 1000))
            self.current_frequency_label.setText(f"Current Frequency: {self.sdr.center_freq / 1e6} MHz")

    def stop_capture(self):
        if self.running:
            self.running = False
            self.timer.stop()
            self.capture_thread.quit()
            self.capture_thread.wait()

    def capture_data(self):
        # This method is no longer used since the data is captured in a separate thread
        pass

    def update_3d_waterfall(self):
        if not self.waterfall_data:
            return

        X = np.arange(self.fft_size)
        Y = np.arange(len(self.waterfall_data)) * self.snapshot_duration
        Z = np.array(self.waterfall_data)

        X, Y = np.meshgrid(X, Y)

        self.canvas_3d.figure.clf()
        ax = self.canvas_3d.figure.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=self.current_cmap)

        ax.set_xlabel('Frequency Bins')
        ax.set_ylabel('Time (s)')
        ax.set_zlabel('Power Spectrum (dB)')
        ax.set_title('3D View')

        self.canvas_3d.draw()

    def update_2d_waterfall(self):
        if not self.waterfall_data:
            return

        self.canvas_2d.figure.clf()
        ax = self.canvas_2d.figure.add_subplot(111)
        ax.imshow(np.array(self.waterfall_data), cmap=self.current_cmap, aspect='auto', origin='lower')

        ax.set_xlabel('Frequency Bins')
        ax.set_ylabel('Time (s)')
        ax.set_title('2D View')

        self.canvas_2d.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaterfallApp()
    window.setWindowTitle("PY-SDR v2.0")
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
