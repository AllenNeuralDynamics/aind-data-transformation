from pathlib import Path
from typing import Union

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class TransformationJobConfig(BaseSettings):
    """Model to define Transformation Job Configs"""

    input_source: Path
    output_directory: Path
    transformation_parameters: Union[BaseSettings, Path]


# TODO: We can probably make this a requests response
class JobResponse(BaseModel):
    message: str
