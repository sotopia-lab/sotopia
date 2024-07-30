# Running Experiments
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

# Hyperparameters that are used in the simulation

## Tags

- `TAG`: The tag of the simulation. This tag is used to identify the simulation in the database.
- `TAG_TO_CHECK_EXISTING_EPISODES`: Scripts like `examples/experiment_eval.py` checks if there are existing episodes with the same tag in the database. If there are, the simulation **will not** be run. This is to avoid running the same simulation twice. If you want to run the simulation again, you can change the tag or set `TAG_TO_CHECK_EXISTING_EPISODES` to `None`.
