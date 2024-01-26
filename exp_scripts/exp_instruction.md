Here are some of the script for running {gpt-3.5-turbo, mixtral-7b-moe} under {normal interaction, omniscient interaction, script generation} mode in {normal, lite} setting.
First run script to generate episodes, like `run_interaction_gpt.sh`, then use `fix_missing_episodes.sh` to fix those error episodes.
You can specify the tags to be fixed in `examples/fix_missing_episodes_with_tag.py`.
If you want to run all modes, you can use the `run_all.sh` and `run_lite_all.sh`
