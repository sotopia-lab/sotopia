cd ..
python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.AGENT1_MODEL="gpt-3.5-turbo"' \
 '--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
 '--gin.BATCH_SIZE=10' \
 '--gin.TAG="interact_gpt-3.5_omniscient"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="interact_gpt-3.5_omniscient"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
 '--gin.OMNISCIENT=True' \
