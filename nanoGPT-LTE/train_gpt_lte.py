"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 nodes with 1 gpu each, example:
- Run on the first (master) node with example IP 10.10.1.1:
$ NCCL_IB_DISABLE=1 torchrun --nnodes=4 --node_rank=0 --master_addr=10.10.1.1 --master_port=5600 train_gpt_lte.py
- Run on the worker node:
$ NCCL_IB_DISABLE=1 torchrun --nnodes=4 --node_rank=2 --master_addr=10.10.1.1 --master_port=5600 train_gpt_lte.py
(NCCL_IB_DISABLE added because cloudlab cluster does not have Infiniband interconnect)
"""

import os
import time
import math
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd

from model import GPTConfig, GPT
from train_util import TrainUtil
import lte
from lte.misc.merge import MergeCondition

# I/O
out_dir = 'out'
eval_batches = 10  # Number of batches to evaluate on
eval_interval = 5  # Interval for wandb logging and checkpointing
log_interval = 1  # Interval for printing iteration loss and time
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
save_checkpoint = False
# model
n_layer = 1
n_head = 1
n_embd = 64
dropout = 0.0      # for pretraining 0 is good, for finetuning try 0.1+
bias = False       # do we use bias inside LayerNorm and Linear layers?
# LTE model settings
wrap_lte = True
freeze_n = 0      # freeze first n transformer blocks
skip_attn = False
skip_logit = True 
skip_mlp = True 
lora_r = 32
lora_alpha = 4096
lte_heads = 32
lte_mode = "mhlora"
lte_merge_steps = 10
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for, (gradually increase lr to max before decay again)
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# wandb logging
wandb_log = False
wandb_project = 'Parallel-Lora'

# data
dataset = 'shakespeare_char'
vocab_size = 65
dtype = 'float16'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 16 # number of contexts to feed. if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 32 # context size
# DDP settings
backend = 'gloo' # nccl threw an error, gloo works fine
# system
device = 'cuda'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
master_process = False
ddp = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
date_str = datetime.now().strftime("%Y-%m-%d")

def check_ddp():
    global master_process
    global ddp
    global gradient_accumulation_steps

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp: 
        print("Running in DDP mode. Rank:", os.environ['RANK'])
        config['ddp'] = True
        dist.init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        if gradient_accumulation_steps % ddp_world_size == 0:
            gradient_accumulation_steps //= ddp_world_size    
    else: 
        master_process = True
        seed_offset = 0 
        ddp_world_size = 1

    torch.manual_seed(1337 + seed_offset)

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"Input tokens processed per iteration will be: {tokens_per_iter}")

def get_lr(it):
    '''
    learning rate decay scheduler (cosine with warmup)

    lr decreases from initial lr gradually to min_lr over lr_decay_iters (generally max_iteration) steps 
    '''
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def create_model(model_args):
    global config 

    if init_from == 'scratch':
        # Create transformer model and print
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
            config[k] = model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        # TODO: Need to use
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
            config[k] = model_args[k]

    # wrap model into DDP container
    model.to(device) 
    if ddp:
        model = DDP(model) # No device_ids because using single GPU

    return model, model_args


def wrap_with_lte(model):
    """
    Replaces some layer weights with frozen weight + lora matrices

    The layers passed as replica layer are skipped during replacement. 

    """
    module = model.module if ddp else model

    # Strict = False because there will be a warning for parameter count mismatch
    # due to weight tying
    # This is okay if we do not replace logit with lora
    model = lte.prepare_model_for_lte(
        model,
        lte.LTEConfig.default(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_heads=lte_heads,
        ),
        mode=lte_mode,
        strict=False,
        use_merge=(lte_merge_steps > 0),

        # Always skip embedding and layer normalization layers
        replica_layers=[module.transformer.wte, module.transformer.wpe, module.transformer.ln_f] 
            + [ module.transformer.h[i].ln_1 for i in range(len(module.transformer.h))]
            + [ module.transformer.h[i].ln_2 for i in range(len(module.transformer.h))]
            + ( [ module.transformer.h[i].attn.c_attn for i in range(len(module.transformer.h))] if skip_attn else [] )
            + ( [ module.transformer.h[i].attn.c_proj for i in range(len(module.transformer.h))] if skip_attn else [] )
            + ( [ module.lm_head] if skip_logit else [] ) 
            + ( [ module.transformer.h[i].mlp.c_fc for i in range(len(module.transformer.h))] if skip_mlp else [] )
            + ( [ module.transformer.h[i].mlp.c_proj for i in range(len(module.transformer.h))] if skip_mlp else [] )
    )

    return model

def create_log_file():
    # Log config 
    config_df = pd.DataFrame([config.values()], columns=config.keys())
    os.makedirs("log", exist_ok=True)
    os.makedirs("log/{}".format(date_str), exist_ok=True)
    config_df.to_csv(f"log/{date_str}/{time_str}_config.csv", index=False)

def train_model(model, model_args, train_util):
    iter_num = 0
    best_val_loss = 1e9

    module = model.module if ddp else model

    # optimizer
    optimizer = module.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    checkpoint = None # free up memory

    run_name = init_from + f'_wrap_lte_{int(wrap_lte)}' + f'_{lte_mode}' \
                + f'_heads_{lte_heads}' + f'_r_{lora_r}' + f'_alpha_{lora_alpha}' 

    # logging
    if wandb_log and master_process:
        import wandb
        wandb_run_name = run_name + "_" + time_str
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    X, Y = train_util.get_batch('train')
    t0 = time.time()
    start_time = time.time()

    log_df = pd.DataFrame(columns=["iter", "train_loss", "val_loss", "train_time"])
    merge_scheduler = MergeCondition(model, merge_steps=lte_merge_steps, method='step')

    while True: 
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Check loss and save checkpoint
        if iter_num % eval_interval == 0 and master_process:
            losses = train_util.estimate_loss()
            print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            train_time = time.time() - start_time
            if wandb_log:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                    'train_time': train_time,
                })
            if (iter_num > 0) and save_checkpoint and (losses['val'] < best_val_loss):
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")

                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            
            # log to two files, because SIGINT can interrupt when writing one
            new_row = pd.DataFrame([[iter_num, losses['train'].item(), 
                losses['val'].item(), train_time]], columns=log_df.columns)
            if len(log_df) > 0:
                log_df = pd.concat([log_df, new_row], ignore_index=True) 
            else: 
                log_df = new_row
            log_df.to_csv(f"log/{date_str}/{time_str}.csv", index=False)
            log_df.to_csv(f"log/{date_str}/{time_str}_2.csv", index=False)

        for micro_step in range(gradient_accumulation_steps):
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
            X, Y = train_util.get_batch('train')
            scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if lte_merge_steps > 0:
            merge_scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        iter_num += 1
        if iter_num >= max_iters:
            break

def generate(model, train_util):
    x, y = train_util.get_batch('val')
    print("x", (''.join(train_util.gpt2_decode(x[0])))[-100:], 
            "\ny", (''.join(train_util.gpt2_decode(y[0])))[-100:])

    gen_y = model.generate(x, 50)
    print("gen_y", (''.join(train_util.gpt2_decode(gen_y[0])))[-300:])

if __name__ == "__main__":
    print("in main")
    check_ddp()
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=vocab_size, dropout=dropout)
    
    # For loading from checkpoint or fine-tuning, take args from the loaded model 
    torch.cuda.empty_cache()
    model, model_args = create_model(model_args)
    
    if wrap_lte:
        model = wrap_with_lte(model)

    print(model)

    train_util = TrainUtil(model, config)
    

    if freeze_n > 0:
        model.freeze_layers(['transformer.wte', 'transformer.wpe'] + [f'transformer.h.{n}' for n in range(freeze_n)])

    # Count number of trainable parameters and their memory
    trainable_params = 0 
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()

    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")
    config['trainable_params'] = trainable_params

    create_log_file()
    
    train_model(model, model_args, train_util)

    generate(model, train_util)
