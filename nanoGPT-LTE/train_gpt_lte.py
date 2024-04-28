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

import torch
import torch.distributed as dist

from model import GPTConfig, GPT
from train_util import TrainUtil
import lte

# I/O
out_dir = 'out'
eval_iters = 1
# model
n_layer = 1
n_head = 1
n_embd = 64
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# data
dataset = 'shakespeare_char'
vocab_size = 65
dtype = 'float16'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # number of contexts to feed. if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 32 # context size
# DDP settings
backend = 'nccl'
# system
device = 'cuda'

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp: 
    print("Running in DDP mode. Rank:", os.environ['RANK'])
    dist.init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else: 
    master_process = True
    seed_offset = 0 
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Input tokens processed per iteration will be: {tokens_per_iter}")

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
iter_num = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# data loader
data_dir = os.path.join('data', dataset)

def create_model():
    # Create transformer model and print
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model)
    return model


def wrap_with_lte(model):
    # Wrap it with LTE and print
    lte.misc.use_custom_attention(model)

    model = lte.prepare_model_for_lte(
        model.cuda(),
        lte.LTEConfig.default(
            lora_r=32,
            lora_alpha=4096,
            num_heads=32,
        ),
        mode="mhlora",
        strict=True,
        replica_layers=[model.transformer.wte, model.transformer.wpe, 
                        model.transformer.h[0].ln_1, model.transformer.h[0].ln_2, 
                        model.transformer.ln_f],
    )

    print(model)
    return model

def train_model(train_util):
    global iter_num
    
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    checkpoint = None # free up memory

    X, Y = train_util.get_batch('train')
    t0 = time.time()

    while True: 
        lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        losses = train_util.estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        for micro_step in range(gradient_accumulation_steps):
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
            X, Y = train_util.get_batch('train')
            scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        iter_num += 1
        if iter_num >= max_iters:
            break



if __name__ == "__main__":
    model = create_model()
    model = wrap_with_lte(model)

    train_util = TrainUtil(model, data_dir, batch_size, block_size, device, eval_iters)

    train_model(train_util)

