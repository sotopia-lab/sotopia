# Benchmark your model as a social agent in Sotopia

```
sotopia benchmark --models <model1> --models <model2> [--only-show-performance]
```
or

```
python sotopia/cli/benchmark/benchmark.py --models <model1> --models <model2> [--only-show-performance]
```
When `only-show-performance` is speficied, only model results with available episodes will be displayed. If this option is not used, the benchmark will be run.
Currently this script would run over 100 simulations on the Sotopia Hard tasks. And the partner model is fixed to be `meta-llama/Llama-3-70b-chat-hf`

An example script is provided in `scripts/display_benchmark_results.sh`

## Using Customized Agents with Benchmark

The default `sotopia benchmark` command uses `LLMAgent` for all agents. If you want to use a customized agent class (e.g., a subclass of `LLMAgent` with custom behavior), you can create your own benchmark script that calls `_benchmark_impl` directly.

Here's an example of how to create a custom benchmark command that uses a customized agent:

```python
import typer
from sotopia.cli.benchmark.benchmark import _benchmark_impl
from sotopia.agents import LLMAgent
from typing import Any, Type
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)


# Define your custom agent class
class CustomSocialWorldModelAgent(LLMAgent):
    """Example custom agent that extends LLMAgent with additional functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Set default social_world_model_name if not provided
        if "social_world_model_name" not in kwargs:
            kwargs["social_world_model_name"] = "gpt-4.1-2025-04-14"
        super().__init__(*args, **kwargs)


@app.command(name="run-custom-benchmark")
def run_custom_benchmark(
    models: Annotated[
        str, typer.Option(help="Comma-separated list of models to benchmark")
    ] = "gpt-4.1-2025-04-14",
    partner_model: Annotated[
        str, typer.Option(help="Partner model to use")
    ] = "gpt-4o-2024-08-06",
    experiment_tag: Annotated[
        str, typer.Option(help="Tag for the benchmark run")
    ] = "custom_agent_trial",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 100,
    push_to_db: Annotated[
        bool, typer.Option(help="Whether to push results to database")
    ] = True,
    evaluator_model: Annotated[
        str, typer.Option(help="Model to use for evaluation")
    ] = "gpt-4o",
    task: Annotated[str, typer.Option(help="Task difficulty level")] = "hard",
) -> None:
    """Run benchmark with custom agent class."""
    # Call _benchmark_impl with your custom agent class
    _benchmark_impl(
        models=models.split(","),
        agent_class=CustomSocialWorldModelAgent,  # Use your custom agent
        partner_model=partner_model,
        evaluator_model=evaluator_model,
        batch_size=batch_size,
        task=task,
        push_to_db=push_to_db,
        tag=experiment_tag,
    )


if __name__ == "__main__":
    app()
```

### Key Points:

1. **Custom Agent Class**: Your custom agent must be a subclass of `LLMAgent` (or another agent class that implements the same interface).

2. **Using `_benchmark_impl`**: The `_benchmark_impl` function accepts an `agent_class` parameter that allows you to specify which agent class to use for the benchmark.

3. **Agent Initialization**: When creating your custom agent, make sure it accepts the same initialization parameters as `LLMAgent` (e.g., `agent_profile`, `model_name`, etc.) and passes them to the parent class.

4. **Running the Custom Benchmark**: Save your script and run it like any other Python script:
   ```bash
   python your_custom_benchmark.py run-custom-benchmark --models gpt-4.1-2025-04-14
   ```

For more information on creating custom agents, see the [Creating your own agents](/concepts/agents#creating-your-own-agents) section.

# Benchmark your model as a evaluator

```
uv run python examples/benchmark_evaluator.py --model=<model> --tag=<tag> --batch-size=<batch_size> --push-to-db
```

This script will re-evaluate the existing episodes with the new model and compare with human annotations.

> **Note:** Sometimes you might need to run the script twice to get the results. This is because the uploading to the database might take some time to complete.

> **Warning:** The re-evaluation does not use the exact same prompt as the original evaluation. However, we have no evidence suggesting that this slight format difference causes any performance discrepancy.
