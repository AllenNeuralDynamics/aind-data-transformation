from abc import ABC, abstractmethod

from models import JobResponse, TransformationJobConfig


class TransformationJob(ABC):

    def __init__(self, job_definition: TransformationJobConfig):
        self.job_configs = job_definition

    @abstractmethod
    def run_job(self) -> JobResponse:
        """Every Job class needs to implement this method"""
        pass
