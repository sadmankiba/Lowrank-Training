# Model (GPT-2)
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 50304
block_size = 512      # instead of 1024
dropout = 0.0
bias = True

# Training
init_from = "scratch"
dataset = "cnn_dailymail"
max_iters = 401
lr_decay_iters = 401
eval_batches = 30 
eval_interval = 10
log_interval = 20
batch_size = 8
gradient_accumulation_steps = 1 # Small GPU memory
weight_decay = 1e-1
warmup_iters = 100

# LTE
wrap_lte = True
lora_r = 32 
lora_alpha = 128 
lte_heads = 4
lte_mode = "dmp"
lte_merge_steps = 10
skip_attn = False
skip_mlp = True
skip_logit = True

# Logging
wandb_log = True
save_checkpoint = False




