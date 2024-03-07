import os
import platform
from pydantic import BaseModel
from pathlib import Path
from datetime import date


def normalize_to_posix_path(path: str):

    if platform.system() == 'Linux':
        return path.replace('\\', '/')

    return path

class FilePathDateInserter(BaseModel):

    filename: Path
    day: date

    def get_filepath(self) -> str:

        # get directory
        directory = self.filename.parent
        current_name = self.filename.name
        date_string = f'{self.day.strftime("%Y_%m_%d")}'

        return directory / f"{date_string}__{current_name}"


class FilepathConverter(BaseModel):

    target_dir: Path  # relative dir if not absolute filepath is not supplied

    def get_filepath(self, filename: str) -> str:
        filename = normalize_to_posix_path(filename)
        filename = str(self.target_dir / filename)

        return filename