# Pretraining BabyGPT is also running into CUDA out of memory :'(

# Model (GPT-2)
n_layer = 6
n_head = 6
n_embd = 384
vocab_size = 50304
batch_size = 16
block_size = 1024
dropout = 0.2
bias = False

# Training
init_from = "scratch"
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




