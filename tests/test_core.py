"""Module to test core and models module"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
SETTINGS_FILE_PATH = TEST_DIR / "core_example_settings" / "settings.json"


class ExampleJobSettings(BasicJobSettings):
    """Create an example job settings class"""

    param: int = 1


class ExampleJob(GenericEtl[ExampleJobSettings]):
    """Create an example etl job class"""

    def run_job(self) -> JobResponse:
        """Required method for all Etl Jobs"""
        return JobResponse(
            status_code=200,
            data=None,
            message=f"Param {self.job_settings.param}",
        )


class TestExampleEtl(unittest.TestCase):
    """Module to test core etl methods"""

    EXAMPLE_ENV_VAR1 = {
        "TRANSFORMATION_JOB_PARAM": "2",
        "TRANSFORMATION_JOB_INPUT_SOURCE": "some_input_dir",
        "TRANSFORMATION_JOB_OUTPUT_DIRECTORY": "some_output_dir",
    }

    @classmethod
    def setUpClass(cls) -> None:
        """Set up tests with basic job settings and etl job"""
        basic_settings = ExampleJobSettings(
            param=2,
            input_source="some_input_dir",
            output_directory="some_output_dir",
        )
        cls.basic_settings = basic_settings
        cls.basic_job = ExampleJob(job_settings=basic_settings)

    def test_settings_with_paths(self):
        """Tests JobSettings can be set with Path types if desired."""
        basic_settings = ExampleJobSettings(
            param=2,
            input_source=Path("some_input_dir"),
            output_directory=Path("some_out_dir"),
        )
        self.assertEqual(Path("some_input_dir"), basic_settings.input_source)
        self.assertEqual(Path("some_out_dir"), basic_settings.output_directory)

    def test_load_cli_args_json_str(self):
        """Tests loading json string defined in command line args"""
        job_settings_json = self.basic_settings.model_dump_json()
        parser = get_parser()
        cli_args = ["--job-settings", job_settings_json]
        parsed_args = parser.parse_args(cli_args)
        loaded_job_settings = ExampleJobSettings.model_validate_json(
            parsed_args.job_settings
        )
        self.assertEqual(self.basic_settings, loaded_job_settings)

    def test_load_cli_args_file_path(self):
        """Tests loading config file defined in command line args"""
        parser = get_parser()
        cli_args = ["--config-file", str(SETTINGS_FILE_PATH)]
        parsed_args = parser.parse_args(cli_args)
        loaded_job_settings = ExampleJobSettings.from_config_file(
            parsed_args.config_file
        )
        self.assertEqual(self.basic_settings, loaded_job_settings)

    @patch.dict(os.environ, EXAMPLE_ENV_VAR1, clear=True)
    def test_load_env_vars(self):
        """Tests loading job settings through env vars"""
        job_settings = ExampleJobSettings()

        self.assertEqual(self.basic_settings, job_settings)

    @patch.dict(os.environ, EXAMPLE_ENV_VAR1, clear=True)
    def test_run_job(self):
        """Tests the run_job method in the ExampleEtl"""
        response = self.basic_job.run_job()
        expected_response = JobResponse(
            status_code=200, data=None, message="Param 2"
        )
        self.assertEqual(response, expected_response)


if __name__ == "__main__":
    unittest.main()
