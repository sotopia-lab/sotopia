gpu_ids=${1:-0}
code_dir=$CODE_DIR/lmlib/train
DATA_DIR=$BASE_DIR/data
CACHE_DIR=$BASE_DIR/cache
SAVE_DIR=$BASE_DIR/save

#alpaca_data=$DATA_DIR/alpaca-cleaned
output_dir=$SAVE_DIR/sasha-lora-llama-13b-v2
#model_name=decapoda-research/llama-7b-hf
model_name=decapoda-research/llama-13b-hf
#model_name=eachadea/vicuna-13b-1.1'
#data_path=$alpaca_data/alpaca_data_cleaned.json
data_path=$DATA_DIR/sotopia/info_sasha.json

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"

# old learning rate 1e-4, lora args use default
python -m lmlib.trainer.train_lora \
    --model_name_or_path $model_name \
    --data_path $data_path \
    --cache_dir $CACHE_DIR \
    --output_dir $output_dir \
    --lora_target_modules 'q_proj,k_proj,v_proj,o_proj' --lora_r 16 \
    --learning_rate 3e-4 \
    --group_by_length \
    --fp16 True \
    --output_dir $output_dir \
    --num_train_epochs 999 --max_steps 500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --logging_steps 1 \
    --save_strategy "steps"\
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048

    #--gradient_checkpointing True \
    #--lazy_preprocess True
    # --fsdp "full_shard auto_wrap offload" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --tf32 True \


    #--resume_from_checkpoint $output_dir/checkpoint-100
