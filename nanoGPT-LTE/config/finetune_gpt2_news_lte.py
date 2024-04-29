# Model (GPT-2)
init_from = "gpt2"

# Training
dataset = "cnn_dailymail"
max_iters = 101
lr_decay_iters = 101
eval_batches = 20 
eval_interval = 10
log_interval = 10
wandb_log = True
gradient_accumulation_steps = 5 * 8
weight_decay = 1e-1

# LTE
wrap_lte = True
lora_r = 32 
lora_alpha = 128 
lte_heads = 4
lte_mode = "mhlora"
lte_merge_steps = 0 

# Logging
wandb_log = True