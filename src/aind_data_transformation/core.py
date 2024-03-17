"""Core abstract class that can be used as a template for etl jobs."""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aind_data_transformation.models import TransformationJobConfig

_T = TypeVar("_T", bound=TransformationJobConfig)


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
        self.job_settings = job_settings

    @abstractmethod
    def run_job(self) -> JobResponse:
        """Abstract method that needs to be implemented by child classes."""
