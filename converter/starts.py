from PyQt6.QtWidgets import QApplication
from gui.file_chooser import FileChooser
import sys


def startGui():
    from gui.gui import App
    app = QApplication(sys.argv)
    w = App()
    sys.exit(app.exec())


def startMerger():
    from misc.file_merger import FileMerger
    app = QApplication(sys.argv)
    chooser = FileChooser()
    file_path = chooser.getPaths(1)
    if chooser.validPaths(1):
        merger = FileMerger()
        merger.merge(file_path)


def startDefaults(app=None, params=None):
    from converter.main import execute
    if not params:
        from converter.movie_utils import MovieOpts
        params = MovieOpts(False)

    print(params)
    if not app:
        app = QApplication(sys.argv)
    chooser = FileChooser()
    file_path, target_path = chooser.getPaths()
    if chooser.validPaths():
        execute(file_path, target_path, params)
