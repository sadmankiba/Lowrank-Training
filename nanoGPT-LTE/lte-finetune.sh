
# Fine-tuning

lte_heads=1 # "1 4"
lora_ranks=768 # "2 8 32 128 768"

# Without LoRA
python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py --wrap_lte=False \
    --max_iters=201 --lr_decay_iters=201 --eval_interval=10

# With LoRA
for lte_head in $lte_heads; do
    for lora_rank in $lora_ranks; do
        python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py \
            --lte_heads=$lte_head --lora_r=$lora_rank \
            --max_iters=201 --lr_decay_iters=201 --eval_interval=10
    done
done