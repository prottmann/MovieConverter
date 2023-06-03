#!/usr/bin/python

from PyQt6.QtWidgets import (QMainWindow, QMessageBox, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QApplication, QCheckBox, QLabel, QComboBox,
                             QLineEdit, QErrorMessage)
from PyQt6.QtGui import QIntValidator

from converter.movie_utils import MovieOpts
from converter.main import execute
from gui.file_chooser import FileChooser
import threading


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.resize(350, 250)
        self.setWindowTitle("Config selection")
        self.center()

        self.tab_widget = Gui()
        self.setCentralWidget(self.tab_widget)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Message", "Are you sure to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


class Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        tab = QTabWidget()
        t1 = QWidget()
        t1.setLayout(self.initSettings())
        tab.addTab(t1, "Settings")

        t2 = QWidget()
        t2.setLayout(self.initEndings())
        tab.addTab(t2, "Endings")
        layout = QVBoxLayout()
        layout.addWidget(tab)

        layout.addLayout(self.initStartQuit())

        self.setLayout(layout)

    def initStartQuit(self):
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        sbtn = QPushButton("Start")
        sbtn.clicked.connect(self.parseConfig)
        qbtn = QPushButton("Quit")
        qbtn.clicked.connect(QApplication.instance().quit)
        hbox.addWidget(sbtn)
        hbox.addWidget(qbtn)
        return hbox

    def initEndings(self):
        self.endings = [".mkv", ".ts", ".mp4", ".mov", ".avi"]
        self.boxes = []
        vbox = QVBoxLayout()
        for e in self.endings:
            self.boxes.append(QCheckBox(e))
            vbox.addWidget(self.boxes[-1])
            self.boxes[-1].setChecked(e == ".mkv" or e == ".ts")
        return vbox

    def initSettings(self):
        vbox = QVBoxLayout()
        # vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Encoder:"))
        self.enc = QComboBox()
        self.enc.addItems(["x264", "x265", "svt-av1"])
        self.enc.currentIndexChanged.connect(self.preparePresets)
        hbox.addWidget(self.enc)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Preset:"))
        self.preset = QComboBox()
        self.preset.addItems(["slow", "medium", "fast"])
        hbox.addWidget(self.preset)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        self.delete = QCheckBox("Delete original file")
        self.delete.setChecked(True)
        hbox.addWidget(self.delete)
        vbox.addLayout(hbox)
        hbox.addStretch(1)

        hbox = QHBoxLayout()
        self.stereo = QCheckBox("Add only stereo track")
        hbox.addWidget(self.stereo)
        vbox.addLayout(hbox)
        hbox.addStretch(1)

        hbox = QHBoxLayout()
        self.shutdown = QCheckBox("Shutdown after completion")
        hbox.addWidget(self.shutdown)
        vbox.addLayout(hbox)
        hbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Crf Value [0-60]:"))
        self.line = QLineEdit("20")
        self.line.setValidator(QIntValidator(0, 60))
        hbox.addWidget(self.line)
        vbox.addLayout(hbox)
        hbox.addStretch(1)
        return vbox

    def parseConfig(self):
        params = MovieOpts()
        params.quality = self.line.text()
        params.preset = self.preset.currentText()
        params.delete_files = self.delete.isChecked()
        params.shutdown = self.shutdown.isChecked()
        params.replace_stereo = self.stereo.isChecked()
        params.endings = []
        for e, v in zip(self.endings, self.boxes):
            if v.isChecked():
                params.endings.append(e)
        print(params)
        if len(params.endings) == 0:
            print("No File-endings given!")
            error_dialog = QErrorMessage()
            error_dialog.setWindowTitle("Missing Ending!")
            error_dialog.showMessage("No File-endings given!")
            error_dialog.exec()
            return
        chooser = FileChooser()
        file_path, target_path = chooser.getPaths()
        if chooser.validPaths():
            thread = threading.Thread(target=execute,
                                      args=(file_path, target_path, params))
            thread.start()
            # execute(file_path, target_path, params)

    def preparePresets(self):
        if "x" in self.enc.currentText():
            options = ["slow", "medium", "fast"]
        else:
            options = ["5", "6", "7"]

        for i, v in enumerate(options):
            self.preset.setItemText(i, v)
