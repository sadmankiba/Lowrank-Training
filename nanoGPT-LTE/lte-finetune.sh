#######
# Experiment for GPT-2 + LTE results  
#######

# Config
max_iters=201
lr_decay_iters=201
eval_interval=10
learning_rate=3e-3
min_lr=3e-4

# Sweep 
lte_heads="1 4"
lora_ranks="1 4 16 64"

finetune_gpt2() {
    python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py --wrap_lte=False \
        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
        --eval_interval=$eval_interval \
        --learning_rate=$learning_rate --min_lr=$min_lr
}

finetune_mhlora() {
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py \
                --lte_heads=$lte_head --lora_r=$lora_rank \
                --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters --eval_interval=$eval_interval
        done
    done
}

finetune_dmp_nomerge() {
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py \
                --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=10 \
                --skip_logit=True --skip_mlp=True  \
                --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                --eval_interval=$eval_interval --learning_rate=$learning_rate
        done
    done
}


finetune_gpt2 
finetune_dmp_nomerge