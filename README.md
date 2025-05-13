## Get started

### Install locally
We recommend using a virtual environment, e.g. with anaconda3: `conda create -n sotopia python=3.11; conda activate sotopia;`.

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
