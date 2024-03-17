from pathlib import Path

from pydantic_settings import BaseSettings


class TransformationJobConfig(BaseSettings):
    """Model to define Transformation Job Configs"""

    input_source: Path
    output_directory: Path
