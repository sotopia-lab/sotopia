# sotopia
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Github Action](https://github.com/ProKil/web-nav-env/actions/workflows/tests.yml/badge.svg?branch=main)]()
[![Github Action](https://github.com/ProKil/web-nav-env/actions/workflows/pre-commit.yml/badge.svg?branch=main)]()

The code repo for project an Open-ended Social Learning Environment

## Installation

This package supports Python 3.11 and above. We recommend using a virtual environment to install this package, e.g. with anaconda3: `conda create -n sotopia python=3.11; conda activate sotopia`. Then, install the requirements and this package.
```
pip install -r requirements.txt; pip install -e .
```

## Contribution
### Install dev options
```bash
pip install -e ".[dev]"
mypy --install-types --non-interactive browser_env
pip install pre-commit
pre-commit install
```
### New branch for each feature
`git checkout -b feature/feature-name` and PR to `main` branch.
### Before committing
Run `pytest` to make sure all tests pass (this will ensure dynamic typing passed with beartype) and `mypy --strict .` to check static typing.
### Check github action result
Check the github action result to make sure all tests pass. If not, fix the errors and push again.