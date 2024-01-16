cd ../sotopia
python examples/generate_script.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch_script.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.SCRIPT_MODEL="gpt-3.5-turbo"' \
 '--gin.AGENT1_MODEL="gpt-3.5-turbo"' \
 '--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
 '--gin.BATCH_SIZE=10' \
 '--gin.TAG="script_full_gpt3.5_gpt3.5_rewrite_lite"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=True' \
