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

### Install locally
We recommend using a virtual environment, e.g. with anaconda3: `conda create -n sotopia python=3.11; conda activate sotopia; curl -sSL https://install.python-poetry.org | python3`.

Then:
`python -m pip install sotopia; sotopia install`

This will setup the necessary environment variables and download the necessary data.

> [!TIP]
> Having trouble installing? Or don't want to install redis for now? We are working on a public redis server for you to use. Stay tuned!

OpenAI key is required to run the code. Please set the environment variable `OPENAI_API_KEY` to your key. The recommend way is to add the key to the conda environment:
```bash
conda env config vars set OPENAI_API_KEY=your_key
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
            "agent1": "gpt-4o-mini",
            "agent2": "gpt-4o-mini",
        },
        sampler=UniformSampler(),
    )
)
```
or run
```bash
python examples/minimalist_demo.py
```
