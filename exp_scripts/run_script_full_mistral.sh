cd ..
python examples/generate_script.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch_script.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.SCRIPT_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"' \
 '--gin.AGENT1_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"' \
 '--gin.AGENT2_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"' \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="script_full_mistral_gpt3.5_rewrite"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
