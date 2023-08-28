python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.AGENT1_MODEL="togethercomputer/llama-2-70b-chat"' \
 '--gin.AGENT2_MODEL="gpt-4"' \
 '--gin.BATCH_SIZE=20' \
 '--gin.TAG="aug20_gpt4_llama-2-70b-chat_zqi2"' \
 '--gin.PUSH_TO_DB=True'
