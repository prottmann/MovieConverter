from gui.file_chooser import FileChooser
import os


class FileMerger(object):
    """docstring for FileMerger"""

    def __init__(self):
        super(FileMerger, self).__init__()

    def merge(self, file_path, ending=".trp"):
        files = []
        for name in os.listdir(file_path):
            fullPath = os.path.join(file_path, name)

            if fullPath.endswith(ending) and name.startswith("00"):
                files.append(fullPath)
        files_path = os.path.join(file_path, "files.txt")
        path_merged = os.path.join(file_path, "merged.ts")
        with open(files_path, "w") as f:
            for file in sorted(files):
                f.write(f"file '{file}'\n")

        command = f"ffmpeg -safe 0 -f concat -i \"{files_path}\" -c copy -map 0 {path_merged}"

        print(command)
        os.system(command)