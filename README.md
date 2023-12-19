# MLxplain ![build](https://github.com/HES-XPLAIN/mlxplain/actions/workflows/build.yml/badge.svg)
An open platform for accelerating the development of eXplainable AI systems

* [Documentation](https://hes-xplain.github.io/mlxplain/docs/)
* [Coverage](https://hes-xplain.github.io/mlxplain/cov/)

## Contribution

### Install dependencies

#### Using pip

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

#### Using Rye

- Install [Rye](https://rye-up.com/) and [add shims](https://rye-up.com/guide/installation/) to your PATH.

Ensure `rye` is accessible in the `$PATH` environment variable.
Rye will automatically download the suitable Python toolchain as needed.

To check the installation, check the following commands return an output:

```shell
rye --version
```

Install python dependencies and activate the virtualenv:

```shell
rye sync
rye shell
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

To publish the package on [PyPI](https://pypi.org/project/mlxplain/), refer to [RELEASE](RELEASE.md).
