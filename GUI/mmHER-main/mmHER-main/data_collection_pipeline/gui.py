import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout,
    QVBoxLayout, QSlider, QFrame, QGraphicsView, QGraphicsScene, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt

class VitalSignGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vital Sign Monitoring IWR6843ISK-ODS")
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
        """)
        self.initUI()

    def initUI(self):
        # === Left Panel ===
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)

        left_layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding))

        title_label = QLabel("Vital Sign\nMonitoring\nIWR6843ISK-ODS")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        left_layout.addSpacing(20)

        # COM Input
        com_layout = QHBoxLayout()
        com_label = QLabel("COM :")
        self.com_input = QLineEdit()
        self.com_input.setFixedWidth(100)
        com_layout.addWidget(com_label)
        com_layout.addWidget(self.com_input)
        left_layout.addLayout(com_layout)

        # Duration Input
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Capture Duration (s) :")
        self.duration_input = QLineEdit()
        self.duration_input.setFixedWidth(100)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_input)
        left_layout.addLayout(duration_layout)

        left_layout.addSpacing(20)

        # Start Button
        self.start_button = QPushButton("START CAPTURE")
        left_layout.addWidget(self.start_button)

        left_layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # === Right Panel ===
        right_layout = QVBoxLayout()

        # HR Display
        self.hr_label = QLabel("HR: ___ BPM")
        right_layout.addWidget(self.hr_label)
        right_layout.addWidget(QLabel("Heart Rate Waveform"))

        self.hr_waveform = QGraphicsView()
        self.hr_waveform.setFixedHeight(100)
        self.hr_scene = QGraphicsScene()
        self.hr_waveform.setScene(self.hr_scene)
        right_layout.addWidget(self.hr_waveform)

        # HR Frame slider
        hr_frame_layout = QHBoxLayout()
        self.hr_frame_label = QLabel("Frame __/__")
        self.hr_slider = QSlider(Qt.Horizontal)
        hr_frame_layout.addWidget(self.hr_frame_label)
        hr_frame_layout.addWidget(self.hr_slider)
        right_layout.addLayout(hr_frame_layout)

        # RR Display
        self.rr_label = QLabel("RR: ___ BPM")
        right_layout.addWidget(self.rr_label)
        right_layout.addWidget(QLabel("Respiratory Rate Waveform"))

        self.rr_waveform = QGraphicsView()
        self.rr_waveform.setFixedHeight(100)
        self.rr_scene = QGraphicsScene()
        self.rr_waveform.setScene(self.rr_scene)
        right_layout.addWidget(self.rr_waveform)

        # RR Frame slider
        rr_frame_layout = QHBoxLayout()
        self.rr_frame_label = QLabel("Frame __/__")
        self.rr_slider = QSlider(Qt.Horizontal)
        rr_frame_layout.addWidget(self.rr_frame_label)
        rr_frame_layout.addWidget(self.rr_slider)
        right_layout.addLayout(rr_frame_layout)

        # === Combine Panels ===
        main_layout = QHBoxLayout()
        left_widget = QFrame()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(320)

        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VitalSignGUI()
    gui.resize(920, 520)
    gui.show()
    sys.exit(app.exec_())
