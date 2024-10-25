"""Core abstract class that can be used as a template for etl jobs."""

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PathLike = TypeVar("PathLike", str, Path)


def get_parser() -> argparse.ArgumentParser:
    """
    Get a standard parser that can be used to parse command line args
    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--job-settings",
        required=False,
        type=str,
        help=(
            r"""
            Instead of init args the job settings can optionally be passed in
            as a json string in the command line.
            """
        ),
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=False,
        type=Path,
        help=(
            r"""
            Instead of init args the job settings can optionally be loaded from
            a config file.
            """
        ),
    )
    return parser


class BasicJobSettings(BaseSettings):
    """Model to define Transformation Job Configs"""

    model_config = SettingsConfigDict(env_prefix="TRANSFORMATION_JOB_")
    input_source: PathLike
    output_directory: PathLike

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


_T = TypeVar("_T", bound=BasicJobSettings)


class JobResponse(BaseModel):
    """Standard model of a JobResponse."""

    model_config = ConfigDict(extra="forbid")
    status_code: int
    message: Optional[str] = Field(None)
    data: Optional[str] = Field(None)


class GenericEtl(ABC, Generic[_T]):
    """A generic etl class. Child classes will need to create a JobSettings
    object that is json serializable. Child class will also need to implement
    the run_job method, which returns a JobResponse object."""

    def __init__(self, job_settings: _T):
        """
        Class constructor for the GenericEtl class.
        Parameters
        ----------
        job_settings : _T
          Generic type that is bound by the BaseSettings class.
        """
        self.job_settings = job_settings.model_copy(deep=True)
        # Parse str into Paths
        if isinstance(self.job_settings.input_source, str):
            self.job_settings.input_source = Path(
                self.job_settings.input_source
            )
        if isinstance(self.job_settings.output_directory, str):
            self.job_settings.output_directory = Path(
                self.job_settings.output_directory
            )

    @abstractmethod
    def run_job(self) -> JobResponse:
        """Abstract method that needs to be implemented by child classes."""
