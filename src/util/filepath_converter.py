from pydantic import BaseModel
from pathlib import Path
from datetime import date


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

        parent_dir = Path(filename).parent

        # check validity of path
        assert len(str(parent_dir)) > 1, "accessing the root directory is not allowed"

        return filename if parent_dir.exists() else str(self.target_dir / filename)