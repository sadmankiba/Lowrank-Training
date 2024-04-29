import os 

import torch
import numpy as np
import pandas as pd
import tiktoken


class TrainUtil:
    def __init__(self, model, config):
        self.model = model
        self.batch_size = config["batch_size"]
        self.block_size = config["block_size"]
        self.data_dir = os.path.join('data', config["dataset"])
        self.device = config["device"]
        self.eval_batches = config["eval_batches"]

    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:(i + self.block_size)]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:(i + 1 + self.block_size)]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_batches)
            for k in range(self.eval_batches):
                X, Y = self.get_batch(split)
                
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        
        self.model.train()
        return out
    
    def gpt2_decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().tolist()
        
        enc = tiktoken.encoding_for_model('gpt2')
        return enc.decode(x)
