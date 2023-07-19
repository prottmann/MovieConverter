from dataclasses import dataclass, field


@dataclass(init=True)
class Parameters(object):
    """docstring for Parameters"""
    quality: int = 20
    preset: str = "slow"
    codec: str = "x264"
    delete_files: bool = False
    detect_logo: bool = False
    shutdown: bool = False
    replace_stereo: bool = False
    keep_audio: bool = False
    endings: list[str] = field(default_factory=list)

    def __init__(self, read=False):
        if read:
            self.readYaml()

        self.endings = [".mkv", ".ts"]

    def readYaml(self) -> None:
        """Read yaml file and fill attributes"""
        import yaml
        with open("converter/config.yaml") as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            p = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in p.items():
                setattr(self, key, value)
                # self.[key] = p[key]
