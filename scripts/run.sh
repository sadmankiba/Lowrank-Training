# Run for DDP with 2 nodes
NCCL_IB_DISABLE=1 torchrun --nnodes=2 --node_rank=1 --master_addr=10.10.1.1 --master_port=5600 train_gpt_lte.py

# Run single with config
python3 train_gpt_lte.py config/train_shakespeare_char_lte.py

# Run for DDP with 4 nodes vanilla nanoGPT
torchrun --nnodes=4 --node_rank=1 --master_addr=10.10.1.1 --master_port=5600 train_gpt_lte.py config/train_shakespeare_char_lte.py --wrap_lte=False

# ===== Fine Tuning with GPT-2 model ======

# Pre-load GPT-2 model from huggingface and dataset cnn_dailymail for fine-tuning
python3 train_gpt_lte.py --init_from=gpt2 --dataset=cnn_dailymail --eval_batches=20 --wrap_lte=False

# Train for 10 iterations
<> --max_iters=10 --eval_batches=20 --eval_interval=2

# Freeze 
<> --freeze_n=5

# ===== Pre-training with GPT-2 config and LTE =====

# MH LoRA 
python3 train_gpt_lte.py config/train_gpt2_news_lte.py

# ===== Fine-tuning with GPT-2 config and LTE =====

# MH LoRA
python3 train_gpt_lte.py config/finetune_gpt2_news_lte.py
