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
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Sotopia%20Demo-orange)](https://demo.sotopia.world)



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



## Get started

### Install locally
We recommend using a virtual environment, e.g. with uv: `pip install uv; uv sync --all-extras`.

> [!NOTE]
> You can of course use any other package manager to install the dependencies (e.g. pip, conda, etc.). But we strongly recommend using uv, especially for the development of Sotopia.


Then:
`uv run sotopia install`

### Storage Backend Options

Sotopia supports two storage backends:

1. **Redis (default)** - Recommended for production use
   - Requires Redis server running
   - We recommend using Docker: `docker run -d -p 6379:6379 redis/redis-stack-server:latest`
   - Set via: `export SOTOPIA_STORAGE_BACKEND=redis` (or leave unset)

2. **Local JSON** - Simpler setup for development/testing
   - No external dependencies required
   - Stores data in `~/.sotopia/data/`
   - Set via: `export SOTOPIA_STORAGE_BACKEND=local`
   - **Note**: Experimental framework features require Redis

> [!WARNING]
> For Redis setup, we recommend using Docker. Other installation methods have been shown to be error-prone.

### Environment Variables

Sotopia uses environment variables for configuration. The recommended way to set them is using a `.env` file in the project root:

```bash
# Create a .env file
cat > .env << EOF
# Required: OpenAI API key
OPENAI_API_KEY=your_openai_key_here

# Storage backend: "redis" (default) or "local"
SOTOPIA_STORAGE_BACKEND=local

# Redis connection (only needed if using Redis backend)
# REDIS_OM_URL=redis://localhost:6379
EOF
```

**Environment Variables:**
- `OPENAI_API_KEY` (required): Your OpenAI API key for running LLM-based simulations
- `SOTOPIA_STORAGE_BACKEND` (optional): Storage backend - `"redis"` (default) or `"local"`
- `REDIS_OM_URL` (optional): Redis connection string (default: `"redis://localhost:6379"`)

**Loading Environment Variables:**

With `uv`:
```bash
# Option 1: Use --env-file flag
uv run --env-file .env python examples/minimalist_demo.py

# Option 2: Export manually
export $(cat .env | xargs) && uv run python examples/minimalist_demo.py
```

With other tools (pip, conda):
```bash
# Export variables before running
export $(cat .env | xargs)
python examples/minimalist_demo.py
```

> [!NOTE]
> `uv` does not automatically load `.env` files. Use the `--env-file` flag or export variables manually.

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
> [!WARNING]
> You won't be able to run the server locally if you don't have any datasets installed.

or run
```bash
python examples/minimalist_demo.py
```

## Help

> [!IMPORTANT]
> If you are trying to develop on top of Sotopia, we highly recommend to follow the [development guide](https://docs.sotopia.world/contribution/contribution), but cross-reference with this README for latest changes as the documentation may be outdated.
