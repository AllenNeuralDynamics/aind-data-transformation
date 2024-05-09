# aind-data-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.8-blue?logo=python)

## Usage

Please import this package and extend the abstract base class to define a new transformation job
```python
from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
)

# An example JobSettings
class NewTransformJobSettings(BasicJobSettings):
  # Add extra fields needed, for example, a random seed
  random_seed: Optional[int] = 0

# An example EtlJob
class NewTransformJob(GenericEtl[NewTransformJobSettings]):

    # This method needs to be defined
    def run_job(self) -> JobResponse:
        """
        Main public method to run the transformation job
        Returns
        -------
        JobResponse
          Information about the job that can be used for metadata downstream.

        """
        job_start_time = datetime.now()
        # Do something here
        job_end_time = datetime.now()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time-job_start_time}",
            data=None,
        )

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

