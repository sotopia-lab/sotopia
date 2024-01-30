python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.SCRIPT_MODEL="gpt-3.5-turbo-finetuned"' \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="finetuned_eval_full"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="finetuned_eval_full"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=True' \
