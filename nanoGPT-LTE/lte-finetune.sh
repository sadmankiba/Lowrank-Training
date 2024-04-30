
# Fine-tuning

lte_heads="1 4"
lora_ranks="1 2 4 8 16 32 64 128"

# Without LoRA
python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py --wrap_lte=False \
    --max_iters=401 --lr_decay_iters=401 --eval_interval=20

# With LoRA
for lte_head in $lte_heads; do
    for lora_rank in $lora_ranks; do
        python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py \
            --lte_heads=$lte_head --lora_r=$lora_rank \
            --max_iters=201 --lr_decay_iters=201 --eval_interval=10
    done
done