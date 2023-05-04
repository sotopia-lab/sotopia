gpu_ids=${1:-0}
code_dir=$CODE_DIR/alpaca-lora
DATA_DIR=$BASE_DIR/data
CACHE_DIR=$BASE_DIR/cache
SAVE_DIR=$BASE_DIR/save

alpaca_data=$DATA_DIR/alpaca-cleaned
output_dir=$SAVE_DIR/lora-alpaca
# model_name=decapoda-research/llama-7b-hf
model_name=decapoda-research/llama-13b-hf
#model_name=chavinlo/alpaca-native
#model_name=eachadea/vicuna-13b-1.1
data_path=$alpaca_data/alpaca_data_cleaned.json
lora_weight_path=$SAVE_DIR/sasha-lora-llama-13b-v2

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

echo model name $model_name
echo cache dir $CACHE_DIR
python -m lmlib.serve.cli --model_path $model_name --cache_dir $CACHE_DIR --lora_weight_path $lora_weight_path --load_8bit
