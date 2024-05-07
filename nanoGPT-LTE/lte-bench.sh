#######
# Experiment for GPT-2 + LTE results  
#######

# Config
max_iters=501
lr_decay_iters=501
eval_interval=10
learning_rate=6e-4
min_lr=6e-5

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

finetune_dmp_merge() {
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py \
                --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=10 \
                --skip_logit=True --skip_mlp=True  \
                --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                --eval_interval=$eval_interval \
                --learning_rate=$learning_rate --min_lr=$min_lr
        done
    done
}

pretrain_gpt2() {
    python3 train_gpt_lte.py config/train_gpt2_news_lte.py --wrap_lte=False \
        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
        --eval_interval=$eval_interval \
        --learning_rate=$learning_rate --min_lr=$min_lr
}

pretrain_dmp_merge() {
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=10 \
                --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                --eval_interval=$eval_interval \
                --learning_rate=$learning_rate --min_lr=$min_lr
        done
    done
}

pretrain_mhlora_nomerge() {
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="mhlora" \
                --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=0 \
                --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                --eval_interval=$eval_interval \
                --learning_rate=$learning_rate --min_lr=$min_lr
        done
    done
}

pretrain_dmp_merge_sweep1() {
    lte_heads="2 4 8 16"
    lora_ranks="1 4 16 64"
    skip_mlp_logits="True False"
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            for skip_mlp_logit in $skip_mlp_logits; do
                python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                    --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                    --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=10 \
                    --skip_mlp=$skip_mlp_logit --skip_logit=$skip_mlp_logit \
                    --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                    --eval_interval=$eval_interval \
                    --learning_rate=$learning_rate --min_lr=$min_lr
            done
        done
    done
}

pretrain_dmp_merge_sweep2() {
    lte_heads="8"
    lora_ranks="4 64"
    lora_alpha_ms=("1" "4" "16")
    lora_alpha_ds=("4" "1" "1")
    skips=("--skip_attn=False --skip_mlp=False --skip_logit=True" \
             "--skip_attn=False --skip_mlp=True --skip_logit=False" \
             "--skip_attn=True --skip_mlp=False --skip_logit=False")
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            
                for alpha_index in ${!lora_alpha_ms[@]}; do
                    python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                        --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                        --lora_alpha=$((lora_alpha_ms[$alpha_index] * lora_rank / lora_alpha_ds[$alpha_index])) \
                        --lte_merge_steps=10 \
                        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                        --skip_attn=False --skip_mlp=False --skip_logit=True \
                        --eval_interval=$eval_interval \
                        --learning_rate=$learning_rate --min_lr=$min_lr
                    
                    python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                        --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                        --lora_alpha=$((lora_alpha_ms[$alpha_index] * lora_rank / lora_alpha_ds[$alpha_index])) \
                        --lte_merge_steps=10 \
                        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                        --skip_attn=False --skip_mlp=True --skip_logit=False \
                        --eval_interval=$eval_interval \
                        --learning_rate=$learning_rate --min_lr=$min_lr
                    
                    python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                        --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                        --lora_alpha=$((lora_alpha_ms[$alpha_index] * lora_rank / lora_alpha_ds[$alpha_index])) \
                        --lte_merge_steps=10 \
                        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                        --skip_attn=True --skip_mlp=False --skip_logit=False \
                        --eval_interval=$eval_interval \
                        --learning_rate=$learning_rate --min_lr=$min_lr
                done
            
        done
    done
}

pretrain_dmp_merge_sweep3() {
    lte_heads="4 16"
    lora_ranks="8"
    merge_steps="10 20 30 40"
    for lte_head in $lte_heads; do
        for lora_rank in $lora_ranks; do
            for merge_step in $merge_steps; do 
                python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                    --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                    --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=$merge_step \
                    --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                    --eval_interval=$eval_interval \
                    --learning_rate=$learning_rate --min_lr=$min_lr
            done
        done
    done
}

pretrain_dmp_merge_replacements() {
    lte_head="4"
    lora_ranks="1 4 16 64"
    skips="True False"

    for lora_rank in $lora_ranks; do
        for skip_attn in $skips; do
            for skip_mlp in $skips; do
                for skip_logit in $skips; do
                    python3 train_gpt_lte.py config/train_gpt2_news_lte.py \
                        --lte_heads=$lte_head --lora_r=$lora_rank --lte_mode="dmp" \
                        --lora_alpha=$((4 * lora_rank)) --lte_merge_steps=10 \
                        --skip_attn=$skip_attn --skip_mlp=$skip_mlp --skip_logit=$skip_logit \
                        --max_iters=$max_iters --lr_decay_iters=$lr_decay_iters \
                        --eval_interval=$eval_interval \
                        --learning_rate=$learning_rate --min_lr=$min_lr
                done
            done
        done
    done
}

pretrain_dmp_merge_replacements