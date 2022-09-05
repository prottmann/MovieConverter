from PyQt6.QtWidgets import QFileDialog, QWidget
import os


class FileChooser(QWidget):
    """docstring for FileChooser"""
    file_path = None
    target_path = None

    def __init__(self):
        super(FileChooser, self).__init__()

    def getPaths(self, number_of_paths: int = 2):
        self.file_path = QFileDialog.getExistingDirectory(
            self, "Choose directory with input movies", os.getenv("HOME"))
        print(f"Input  path: {self.file_path}")

        if number_of_paths > 1:
            self.target_path = QFileDialog.getExistingDirectory(
                self, "Choose directory for output movies", os.getenv("HOME"))
            print(f"Output path: {self.target_path}")
            return self.file_path, self.target_path
        return self.file_path

    def validPaths(self, number_of_paths: int = 2):
        if not self.file_path or (number_of_paths > 1 and not self.target_path):
            print("No path")
            return False
        else:
            return True
