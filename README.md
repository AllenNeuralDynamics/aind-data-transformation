# aind-data-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)

## Usage

There are 4 main ways to run a data transformation job:
- from a python script
- from the command line passing in the settings as a json string
- from the command line pointing to a config file
- from the command line with env vars

Assuming `output_dir` exists:

### From python
```python
from aind_data_transformation.ephys.ephys_job import EphysJobSettings, EphysCompressionJob
from pathlib import Path

input_source = Path("./tests/resources/v0.6.x_neuropixels_multiexp_multistream")
output_dir = Path("output_dir")

job_settings = EphysJobSettings(input_source=input_source, output_directory=output_dir)
job = EphysCompressionJob(job_settings=job_settings)

response = job.run_job()
```

### From the command line passing in settings as a json str
```bash
python -m aind_data_transformation.ephys.ephys_job --job-settings '{"input_source":"./tests/resources/v0.6.x_neuropixels_multiexp_multistream","output_directory":"output_dir"}'
```

### From the command line passing in settings via a config file
```bash
python -m aind_data_transformation.ephys.ephys_job --config-file configs.json
```

### From the command line passing in settings via environment variables
```bash
export TRANSFORMATION_JOB_INPUT_SOURCE="./tests/resources/v0.6.x_neuropixels_multiexp_multistream"
export TRANSFORMATION_JOB_OUTPUT_DIRECTORY="output_dir"
python -m aind_data_transformation.ephys.ephys_job
```


## Contributing

The development dependencies can be installed with
```bash
pip install -e .[dev]
```

### Adding a new transformation job
Any new job needs a settings class that inherits the BasicJobSettings class. This requires the fields input_source and output_directory and makes it so that the env vars have the TRANSFORMATION_JOB prefix.

Any new job needs to inherit the GenericEtl class. This requires that the main public method to execute is called `run_job` and returns a JobResponse.

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

