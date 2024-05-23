# Example Scripts For Using The Library

## Example 1: Evaluating existing episodes

```python
python examples/evaluate_existing_episodes.py --tag=<tag to upload to the database> --model=<the model used to re-evaluate the existing episodes> --batch_size=<batch size used for evaluation> --push-to-db
```

Run ```python examples/evaluate_existing_episodes.py --help``` for more information.

### Example 1.1: benchmarking the evaluator
```python
python examples/benchmark_evaluator.py --push-to-db --model=<the model used to be evaluated as evaluator> --tag=<tag to upload to the database> --batch_size=10
```

## Example 2: Generate script-like episodes
See `docs/simulation_modes.md` for more information.

## Example 3: Benchmarking the models as social agents

```Bash
EVAL_MODEL="gpt-4o-2024-05-13"
python examples/benchmark_social_agents.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.AGENT1_MODEL="groq/llama3-70b-8192"' \
 "--gin.AGENT2_MODEL=${EVAL_MODEL}" \
 '--gin.BATCH_SIZE=10' \
 "--gin.TAG=benchmark_${EVAL_MODEL}" \
 "--gin.TAG_TO_CHECK_EXISTING_EPISODES=benchmark_${EVAL_MODEL}" \
 '--gin.PUSH_TO_DB=True' \
 '--gin.OMNISCIENT=False' \
 '--gin.VERBOSE=False'
```
