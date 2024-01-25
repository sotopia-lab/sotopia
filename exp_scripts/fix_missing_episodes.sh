cd ../sotopia
python examples/fix_missing_episodes_with_tag.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
