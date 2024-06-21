<div style="width: 100%;">
  <img src="figs/title.png" style="width: 100%;" alt="sotopia"></img>
</div>

<h1 align="center">Sotopia: an Open-ended Social Learning Environment</h1>

<div align="center">

[![pypi](https://img.shields.io/pypi/v/sotopia.svg)](https://pypi.python.org/pypi/sotopia)
[![versions](https://img.shields.io/pypi/pyversions/sotopia.svg)](https://github.com/sotopia/sotopia)
[![CI](https://img.shields.io/github/actions/workflow/status/sotopia-lab/sotopia/tests.yml?branch=main&logo=github&label=CI)](https://github.com/sotopia-lab/sotopia/actions?query=branch%3Amain)
[![codecov](https://codecov.io/github/sotopia-lab/sotopia/graph/badge.svg?token=00LRQFX0QR)](https://codecov.io/github/sotopia-lab/sotopia)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14hJOfzpA37PRUzdlFgiqVzUGIhhngqnz?usp=sharing)

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://www.sotopia.world/projects/sotopia)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/abs/2310.11667)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Sotopia%20Dataset-yellow)](https://huggingface.co/datasets/cmu-lti/sotopia)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Sotopia%20Demo-orange)](https://huggingface.co/spaces/cmu-lti/sotopia-space/)



</div>

## News

* [05/2024] Sotopia was presented at ICLR 2024 as a spotlight ⭐!


## Introduction

Sotopia is an open-ended social learning environment that allows agents to interact with each other and the environment. The environment is designed to be a platform for evaluating and faciliating social intelligence in language agents. The environment is designed to be open-ended, meaning that the environment can be easily extended to include new environments and new agents. The environment is also designed to be scalable, meaning that the environment can be easily scaled to include a large number of agents and environments.



```bibtex
@inproceedings{zhou2024sotopia,
  title = {SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents},
  author = {Zhou*, Xuhui and Zhu*, Hao and Mathur, Leena and Zhang, Ruohong and Qi, Zhengyang and Yu, Haofei and Morency, Louis-Philippe and Bisk, Yonatan and Fried, Daniel and Neubig, Graham and Sap, Maarten},
  journal = {ICLR},
  year = {2024},
  url = {https://openreview.net/forum?id=mM7VurbA4r},
}
```

## Help
See [documentation](https://docs.sotopia.world) for more details.


## Get started
### Use on Google Colab

If you want to try it out on Google Colab first, please check out our Colab Tutorial Series:

<ol>
<li> Basic
  <ol>
    <li> <a href="https://colab.research.google.com/github/sotopia-lab/sotopia/blob/main/notebooks/tutorials/1.1-setup.ipynb">1.1 Playing with Sotopia in a Google Colab Notebook</a> </li>
    <li> <a href="https://colab.research.google.com/github/sotopia-lab/sotopia/blob/main/notebooks/tutorials/1.2-browse-data.ipynb">1.2 Browsing sotopia data</a> </li>
  </ol>
</li>
<li>
Building your own social agent (coming soon!)
</li>
</ol>

### Install locally
This package supports Python 3.11 and above. In one line,
`pip install sotopia` or `pip install uv; uv pip install sotopia`.

Or from scratch, use a virtual environment, e.g. with anaconda3: `conda create -n sotopia python=3.11; conda activate sotopia; curl -sSL https://install.python-poetry.org | python3`. Then, install the requirements and this package.
```bash
poetry install
```

OpenAI key is required to run the code. Please set the environment variable `OPENAI_API_KEY` to your key. The recommend way is to add the key to the conda environment:
```bash
conda env config vars set OPENAI_API_KEY=your_key
```

For some experiments, TogetherAI key is required to run the code. Please set the environment variable `TOGETHER_API_KEY` to your key. The recommend way is to add the key to the conda environment:
```bash
conda env config vars set TOGETHER_API_KEY=your_key
```

A redis-stack server is required to run the code. Please follow the [instruction](https://redis.io/docs/stack/get-started/install/docker/) to start a redis-stack server or use an existing server. You can also check [Q&A](/docs/troubleshooting.md) to initiate the redis server with the Sotopia data.

The `REDIS_OM_URL` need to be set before loading and saving agents:
```bash
conda env config vars set REDIS_OM_URL="redis://user:password@host:port"
```

Make a folder to store the logs:
```bash
mkdir logs
```



## Easy Sample Server
You can view an episode demo with default parameters with the following:
```python
import asyncio
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

asyncio.run(
    run_async_server(
        model_dict={
            "env": "gpt-4",
            "agent1": "gpt-3.5-turbo",
            "agent2": "gpt-3.5-turbo",
        },
        sampler=UniformSampler(),
    )
)
```
or run
```bash
python examples/minimalist_demo.py
```

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

## Running Experiments
We use `gin-config` to configure the experiments. You don't need to be an expert to use it. The basic syntax is
```bash
python <code_file.py> --gin_file <gin_file1> --gin_file <gin_file2> '--gin.PARAM1=value1' '--gin.PARAM2=value2'
```
The `--gin_file` is used to load and compose the default configuration. The `--gin.PARAM1=value1` is used to overwrite the default configuration. The later configuration will always overwrite the previous one.

Here is an example of running an experiment:

```bash
python examples/experiment_eval.py --gin_file sotopia_conf/generation_utils_conf/generate.gin --gin_file sotopia_conf/server_conf/server.gin --gin_file sotopia_conf/run_async_server_in_batch.gin '--gin.ENV_IDS=["01H7VFHPDZVVCDZR3AARA547CY"]' '--gin.AGENT1_MODEL="gpt-4"' '--gin.BATCH_SIZE=20' '--gin.PUSH_TO_DB=False' '--gin.TAG="test"'
```
For the complete set of parameters, please check the `sotopia_conf` folder.

To run a large batch of environments, you can change the `ENV_IDS` parameter in `sotopia_conf/run_async_server_in_batch.gin` to a list of environment ids. When `gin.ENV_IDS==[]`, all environments on the DB will be used.

## Getting access to your simulation
After running experiments, you can go to the `examples/redis_stats.ipynb` notebook to check the existing episodes (Episode Log section), as well as calculate the performance.

For the original Sotopia simulation in our paper's experiments, you can find how to get them in the [Q&A](/docs/troubleshooting.md) section in the `./docs` folder.

## Adding new characters and environments
You can use the following function with the `**kwargs` being the properties of the `AgentProfile` class. This is the same for the scenarios/environments.
```python
class AgentProfile(JsonModel):
    first_name: str = Field(index=True)
    last_name: str = Field(index=True)
    age: int = Field(index=True, default_factory=lambda: 0)
    occupation: str = Field(index=True, default_factory=lambda: "")
    gender: str = Field(index=True, default_factory=lambda: "")
    gender_pronoun: str = Field(index=True, default_factory=lambda: "")
    public_info: str = Field(index=True, default_factory=lambda: "")
    big_five: str = Field(index=True, default_factory=lambda: "")
    moral_values: list[str] = Field(index=False, default_factory=lambda: [])
    schwartz_personal_values: list[str] = Field(index=False, default_factory=lambda: [])
    personality_and_values: str = Field(index=True, default_factory=lambda: "")
    decision_making_style: str = Field(index=True, default_factory=lambda: "")
    secret: str = Field(default_factory=lambda: "")
    model_id: str = Field(default_factory=lambda: "")

class EnvironmentProfile(JsonModel):
    codename: str = Field(...)
    source: str = Field(...)
    scenario: str = Field(...)
    agent_goals: list[str] = Field(...)
    ...
```

```python

from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile

def add_agent_to_database(**kwargs: dict[str, Any]) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()

def add_env_profile(**kwargs: dict[str, Any]) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()
```
