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

# Benchmark your model as a evaluator

```
uv run python examples/benchmark_evaluator.py --model=<model> --tag=<tag> --batch-size=<batch_size> --push-to-db
```

This script will re-evaluate the existing episodes with the new model and compare with human annotations.

> **Note:** Sometimes you might need to run the script twice to get the results. This is because the uploading to the database might take some time to complete.

> **Warning:** The re-evaluation does not use the exact same prompt as the original evaluation. However, we have no evidence suggesting that this slight format difference causes any performance discrepancy.
