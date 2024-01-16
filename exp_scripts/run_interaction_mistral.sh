cd ../sotopia
python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.AGENT1_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"' \
 '--gin.AGENT2_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"' \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="interact_mistral_moe_7B"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="interact_mistral_moe_7B"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False'
