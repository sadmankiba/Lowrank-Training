# Model (GPT-2)
init_from = "gpt2"

# Training
dataset = "cnn_dailymail"
max_iters = 401
lr_decay_iters = 401
eval_batches = 20 
eval_interval = 20
log_interval = 20
batch_size = 4
gradient_accumulation_steps = 1 # Because our GPU has only 12 GB memory
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
save_checkpoint = False