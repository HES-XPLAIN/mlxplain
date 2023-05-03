# MLxplain ![build](https://github.com/HES-XPLAIN/mlxplain/actions/workflows/build.yml/badge.svg)
An open platform for accelerating the development of eXplainable AI systems

* [Documentation](https://hes-xplain.github.io/mlxplain/docs/)
* [Static analysis](https://hes-xplain.github.io/mlxplain/qodana/)
* [Coverage](https://hes-xplain.github.io/mlxplain/cov/)

## Contribution

### Install Python and Poetry

* Install [Python](https://www.python.org/).
* Install [poetry](https://python-poetry.org/docs/#installation) and add it to your PATH.

Ensure `python` and `poetry` are accessible in the `$PATH` environment variable.

To check the installation, check the following commands return an output:

```shell
python --version
poetry --version
```

Install python dependencies and activate the virtualenv:

```shell
poetry install
poetry shell
```

### Install Pre-commit hooks

Git hooks are used to ensure quality checks are run by all developers every time
before a commit.

```shell
pre-commit install
```

Pre-commit hooks can be run manually with:

```shell
pre-commit run --all-files
```

## Release

To publish the package on [Pypi](https://pypi.org/project/mlxplain/), refer to [RELEASE](RELEASE.md).
