CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 pretraining.py \
    --model_type chatglm2 \
    --model_name_or_path THUDM/chatglm2-6b \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --fp16 \
    --max_train_samples 100 \
    --max_eval_samples 8 \
    --num_train_epochs 0.05 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 5 \
    --evaluation_strategy steps \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 1 \
    --block_size 1024 \
    --output_dir outputs-pt-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True
