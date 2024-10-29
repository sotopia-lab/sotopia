<!-- ![TITLE](figs/title.png) -->
# SocialStream

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)


## Get started

This package supports Python 3.11 and above. We recommend using a virtual environment to install this package, e.g.,

```
conda create -n socialstream python=3.11; conda activate socialstream;  curl -sSL https://install.python-poetry.org | python3
poetry install
```


## Usage
streamlit run app.py

Before that you should specify a Redis database by `conda env config vars set REDIS_OM_URL={YOUR_URL}`. You can also 1. Specify a new database in the sidebar, or 2. Include your temporary scenario in data with suffix `_agents.json` and `_scenarios.json`.


### Chat
First choose the agents (two agents cannot be the same), scenarios and the agent you are going to be, then click `start` to start interaction. When you want to leave and get evaluated, click `stop` to start evaluation.


## Contribution
### Install dev options
```bash
mypy --install-types --non-interactive socialstream
pip install pre-commit
pre-commit install
```
### New branch for each feature
`git checkout -b feature/feature-name` and PR to `main` branch.
### Before committing
Run `pre-commit run --all-files` to run all checks
<!-- Run `pytest` to make sure all tests pass (this will ensure dynamic typing passed with beartype) and `mypy --strict --exclude haicosystem/tools  --exclude haicosystem/grounding_engine/llm_engine_legacy.py .` to check static typing.
(You can also run `pre-commit run --all-files` to run all checks) -->
### Check github action result
Check the github action result to make sure all tests pass. If not, fix the errors and push again.

### About Callbacks in Streamlit
From [Streamlit documentation on session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).
> "When updating Session state in response to events, a callback function gets executed first, and then the app is executed from top to bottom."


In this code, we utilize callbacks to handle the clickability of buttons and ensure that any edited information is promptly and accurately updated.
