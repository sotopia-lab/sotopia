gpu_ids=${1:-0}
code_dir=$CODE_DIR/lmlib/train
DATA_DIR=$BASE_DIR/data
CACHE_DIR=$BASE_DIR/cache
SAVE_DIR=$BASE_DIR/save

#alpaca_data=$DATA_DIR/alpaca-cleaned
output_dir=$SAVE_DIR/lora-vicuna-character
#model_name=decapoda-research/llama-7b-hf
model_name=eachadea/vicuna-13b-1.1
#data_path=$alpaca_data/alpaca_data_cleaned.json
data_path=$DATA_DIR/sotopia/info_emily.json

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

python -m lmlib.trainer.train_lora \
    --base_model $model_name \
    --data_path $data_path \
    --cache_dir $CACHE_DIR \
    --output_dir $output_dir \
    --batch_size 4 \
    --micro_batch_size 1 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint $output_dir/checkpoint-100
