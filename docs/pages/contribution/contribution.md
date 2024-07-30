## Contribution
#### Install dev options
Follow the installation instruction above and then, instead of running `python -m pip install -e .`, run the following commands:

```bash
python -m pip install -e ".[dev]"
mypy --install-types --non-interactive sotopia
python -m pip install pre-commit
pre-commit install
```
#### New branch for each feature
`git checkout -b feature/feature-name` and PR to `main` branch.
#### Before committing
Run `pytest` to make sure all tests pass (this will ensure dynamic typing passed with beartype) and `mypy --strict .` to check static typing.
(You can also run `pre-commit run --all-files` to run all checks)
#### Check github action result
Check the github action result to make sure all tests pass. If not, fix the errors and push again.
