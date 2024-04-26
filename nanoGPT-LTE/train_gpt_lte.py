import os
import time

import torch

from model import GPTConfig, GPT
from train_util import TrainUtil
import lte

# I/O
eval_iters = 1

batch_size = 12 # number of contexts to feed. if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 32 # context size
# model
n_layer = 1
n_head = 1
n_embd = 64
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
vocab_size=100
device = 'cuda'
dataset = 'shakespeare_char'
dtype = 'float16'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
max_iters = 1000

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
iter_num = 0


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

