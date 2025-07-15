import sys
import subprocess
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout,
    QVBoxLayout, QSlider, QFrame, QGraphicsView, QGraphicsScene, QSpacerItem,
    QSizePolicy, QRadioButton, QButtonGroup, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPen

# Dummy data will be replaced with real data after capture
slide_rr = [[], [], [], []]
slide_hr = [[], [], [], []]
hr_vals = [[], [], [], []]
rr_vals = [[], [], [], []]

def process_data():
    global slide_rr, slide_hr, hr_vals, rr_vals
    from processing_script import processing_data_bin
    slide_an1_hr, slide_an2_hr, slide_an3_hr, slide_an4_hr, slide_an1_rr, slide_an2_rr, slide_an3_rr, slide_an4_rr, hr_1, hr_2, hr_3, hr_4, rr_1, rr_2, rr_3, rr_4 = processing_data_bin()
    slide_rr = [slide_an1_rr, slide_an2_rr, slide_an3_rr, slide_an4_rr]
    slide_hr = [slide_an1_hr, slide_an2_hr, slide_an3_hr, slide_an4_hr]
    hr_vals = [hr_1, hr_2, hr_3, hr_4]
    rr_vals = [rr_1, rr_2, rr_3, rr_4]

class CaptureThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, duration):
        super().__init__()
        self.duration = duration

    def run(self):
        try:
            subprocess.run([sys.executable, "data_capture.py", str(self.duration)], check=True)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class VitalSignGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vital Sign Monitoring IWR6843ISK-ODS")
        self.current_frame = 0
        self.selected_antenna = 0
        self.capture_thread = None
        self.setStyleSheet("""
            QWidget {
                background-color: #E5E0D8;
                font-family: 'Segoe UI';
                font-size: 10pt;
                color: #000000;
                font-weight: bold;
            }
            QLabel#Title {
                font-size: 16pt;
                font-weight: bold;
                color: #000000;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #888;
                border-radius: 4px;
                padding: 4px;
                color: #000000;
                font-weight: bold;
            }
            QPushButton {
                background-color: #ffffff;
                color: #000000;
                padding: 6px 12px;
                font-weight: bold;
                border-radius: 6px;
                border: 1px solid #000000;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #aaa;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 14px;
                margin: -4px 0;
                border: 1px solid #333;
            }
            QGraphicsView {
                background-color: #F8F8F8;
                border: 1px solid #aaa;
                border-radius: 4px;
            }
        """
        )
        self.initUI()

    def initUI(self):
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding))

        title_label = QLabel("Vital Sign\nMonitoring\nIWR6843ISK-ODS")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        left_layout.addSpacing(20)

        duration_layout = QHBoxLayout()
        duration_label = QLabel("Capture Duration (s) :")
        self.duration_input = QLineEdit()
        self.duration_input.setFixedWidth(100)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_input)
        left_layout.addLayout(duration_layout)
        left_layout.addSpacing(20)

        self.start_button = QPushButton("START CAPTURE")
        self.start_button.clicked.connect(self.run_capture_script)
        left_layout.addWidget(self.start_button)

        antenna_label = QLabel("Pilih Antena:")
        left_layout.addWidget(antenna_label)
        self.antenna_group = QButtonGroup()
        for i in range(4):
            btn = QRadioButton(f"Antena {i+1}")
            if i == 0:
                btn.setChecked(True)
            self.antenna_group.addButton(btn, i)
            left_layout.addWidget(btn)
        self.antenna_group.buttonClicked.connect(self.update_plot)
        left_layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding))

        right_layout = QVBoxLayout()

        self.hr_label = QLabel("HR: ___ BPM")
        right_layout.addWidget(self.hr_label)
        right_layout.addWidget(QLabel("Heart Rate Waveform"))

        self.hr_waveform = QGraphicsView()
        self.hr_waveform.setFixedHeight(100)
        self.hr_scene = QGraphicsScene()
        self.hr_waveform.setScene(self.hr_scene)
        right_layout.addWidget(self.hr_waveform)

        self.rr_label = QLabel("RR: ___ BPM")
        right_layout.addWidget(self.rr_label)
        right_layout.addWidget(QLabel("Respiratory Rate Waveform"))

        self.rr_waveform = QGraphicsView()
        self.rr_waveform.setFixedHeight(100)
        self.rr_scene = QGraphicsScene()
        self.rr_waveform.setScene(self.rr_scene)
        right_layout.addWidget(self.rr_waveform)

        self.hr_frame_label = QLabel("Frame __/__")
        self.hr_slider = QSlider(Qt.Horizontal)
        self.hr_slider.valueChanged.connect(self.update_plot)

        hr_slider_layout = QHBoxLayout()
        hr_slider_layout.addWidget(self.hr_frame_label)
        hr_slider_layout.addWidget(self.hr_slider)
        right_layout.addLayout(hr_slider_layout)

        main_layout = QHBoxLayout()
        left_widget = QFrame()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(320)
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def run_capture_script(self):
        try:
            duration = int(self.duration_input.text())
            self.clear_plot()
            self.capture_thread = CaptureThread(duration)
            self.capture_thread.finished.connect(self.capture_finished)
            self.capture_thread.error.connect(self.capture_failed)
            self.capture_thread.start()
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Durasi harus berupa angka.")

    def capture_finished(self):
        QMessageBox.information(self, "Capture Selesai", f"Data berhasil direkam.")
        process_data()
        self.hr_slider.setMaximum(len(slide_hr[0]) - 1)
        self.update_plot()

    def capture_failed(self, message):
        QMessageBox.critical(self, "Error", f"Gagal menjalankan capture: {message}")

    def clear_plot(self):
        self.hr_label.setText("HR: ___ BPM")
        self.rr_label.setText("RR: ___ BPM")
        self.hr_frame_label.setText("Frame __/__")
        self.hr_scene.clear()
        self.rr_scene.clear()

    def update_plot(self):
        if len(slide_hr[0]) == 0:
            self.clear_plot()
            return

        self.current_frame = self.hr_slider.value()
        self.selected_antenna = self.antenna_group.checkedId()

        self.hr_frame_label.setText(f"Frame {self.current_frame + 1}/{len(slide_hr[self.selected_antenna])}")
        self.hr_label.setText(f"HR: {hr_vals[self.selected_antenna][self.current_frame]:.1f} BPM")
        self.rr_label.setText(f"RR: {rr_vals[self.selected_antenna][self.current_frame]:.1f} BPM")

        # HR plotting
        self.hr_scene.clear()
        hr_data = slide_hr[self.selected_antenna][self.current_frame]
        height = 100
        center_y = height // 2
        scale_y = 10

        for i in range(len(hr_data) - 1):
            y1 = center_y - hr_data[i] * scale_y
            y2 = center_y - hr_data[i + 1] * scale_y
            self.hr_scene.addLine(i, y1, i + 1, y2, QPen(Qt.black))

        self.hr_scene.setSceneRect(0, 0, len(hr_data), height)

        # RR plotting
        self.rr_scene.clear()
        rr_data = slide_rr[self.selected_antenna][self.current_frame]
        for i in range(len(rr_data) - 1):
            y1 = center_y - rr_data[i] * scale_y
            y2 = center_y - rr_data[i + 1] * scale_y
            self.rr_scene.addLine(i, y1, i + 1, y2, QPen(Qt.darkBlue))

        self.rr_scene.setSceneRect(0, 0, len(rr_data), height)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VitalSignGUI()
    gui.resize(1000, 500)
    gui.show()
    sys.exit(app.exec_())
