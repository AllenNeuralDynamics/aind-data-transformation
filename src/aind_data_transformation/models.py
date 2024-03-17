"""Model for Transformation Job settings"""

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TransformationJobConfig(BaseSettings):
    """Model to define Transformation Job Configs"""

    model_config = SettingsConfigDict(env_prefix="TRANSFORMATION_JOB_")
    input_source: Path
    output_directory: Path

    @classmethod
    def from_config_file(cls, config_file_location: Path):
        """
        Utility method to create a class from a json file
        Parameters
        ----------
        config_file_location : Path
          Location of json file to read.

        """
        with open(config_file_location, "r") as f:
            file_contents = json.load(f)
        return cls.model_validate_json(json.dumps(file_contents))
