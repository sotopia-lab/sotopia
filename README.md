![sotopia](figs/title.png)
# Sotopia: an Open-ended Social Learning Environment
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Github Action](https://github.com/XuhuiZhou/sotopia/actions/workflows/tests.yml/badge.svg?branch=main)]()
[![Github Action](https://github.com/XuhuiZhou/sotopia/actions/workflows/pre-commit.yml/badge.svg?branch=main)]()


## Installation

This package supports Python 3.11 and above. We recommend using a virtual environment to install this package, e.g. with anaconda3: `conda create -n sotopia python=3.11; conda activate sotopia; conda install -c conda-forge pip`. Then, install the requirements and this package.
```bash
python -m pip install -r requirements.txt # make sure the packages are installed in the specific conda environment
python -m pip install -e .
```

OpenAI key is required to run the code. Please set the environment variable `OPENAI_API_KEY` to your key. The recommend way is to add the key to the conda environment:
```bash
conda env config vars set OPENAI_API_KEY=your_key
```

For some experiments, TogetherAI key is required to run the code. Please set the environment variable `TOGETHER_API_KEY` to your key. The recommend way is to add the key to the conda environment:
```bash
conda env config vars set TOGETHER_API_KEY=your_key
```

A redis-stack server is required to run the code. Please follow the [instruction](https://redis.io/docs/stack/get-started/install/docker/) to start a redis-stack server or use an existing server. The `REDIS_OM_URL` need to be set before loading and saving agents:
```bash
conda env config vars set REDIS_OM_URL="redis://user:password@host:port"
```

## Easy Sample Server
You can view an episode demo with default parameters using the following command:
```python
python -m sotopia_conf.server --gin_file="sotopia_conf/server_conf/server.gin" --gin_file="sotopia_conf/generation_utils_conf/generate.gin"
```

## Contribution
### Install dev options
Follow the installation instruction above and then, instead of running `python -m pip install -e .`, run the following commands:

```bash
python -m pip install -e ".[dev]"
mypy --install-types --non-interactive sotopia
python -m pip install pre-commit
pre-commit install
```
### New branch for each feature
`git checkout -b feature/feature-name` and PR to `main` branch.
### Before committing
Run `pytest` to make sure all tests pass (this will ensure dynamic typing passed with beartype) and `mypy --strict .` to check static typing.
(You can also run `pre-commit run --all-files` to run all checks)
### Check github action result
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

### Check the generated episodes
Go to the `examples/redis_stats.ipynb` notebook to check the generated episodes (Episode Log section). As well as calculate the performance.
