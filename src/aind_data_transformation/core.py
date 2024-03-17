"""Core abstract class that can be used as a template for etl jobs."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aind_data_transformation.models import TransformationJobConfig

_T = TypeVar("_T", bound=TransformationJobConfig)


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
