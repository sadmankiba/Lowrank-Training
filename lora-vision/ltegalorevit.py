###################################################################################################
# Comp Sci Big Data - Final Project
# Spring 2024
# Author: Laik Ruetten
# sources: Pulled a lot from galore github and melo github, both papers that we cite in our report
# Date: 04-30-2024
###################################################################################################
import sys

import torch
import torch.nn as nn
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from torch.nn import Linear
import argparse

from peft_pretraining import training_utils, args_utils

import timm
import torch
from lora import LoRA_ViT_timm
import torch.nn.functional as F

import lte

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

from torch.utils.tensorboard import SummaryWriter

###########################################
###########################################
# Hard coded value depending on the dataset used. Not very robust but I am just running tests
num_categories = 9

rank = 8

model = timm.create_model('vit_base_patch16_224', pretrained=True)
#model = LoRA_ViT_timm(vit_model=model, r=rank, alpha=4, num_classes=num_categories)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params/2**20:.3f}M")
num_params = sum(p.numel() for p in model.parameters())
print(f"total parameters: {num_params/2**20:.3f}M")

model = model.cuda()

# converts into an LTE model
model = lte.prepare_model_for_lte(
      model.cuda(),
      lte.LTEConfig.default(
          lora_r=32,
          lora_alpha=4096,
          num_heads=8,
      ),
)

###########################################
# data and model info and parameters
###########################################
NUM_EPOCHS = 1
BATCH_SIZE = 16
lr = 0.005

data_flag = 'pathmnist'
#data_flag = 'breastmnist'
#data_flag = 'chestmnist'
#data_flag = 'pneumoniamnist'
#data_flag = 'breastmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224, mmap_mode='r')
test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224, mmap_mode='r')

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

###########################################

criterion = nn.CrossEntropyLoss()

###########################################
# This function is only here because this code is a hodge podge of different code stuck together.
# It's here to make GaLore work. Interfacing with the command line for actual arguments like 
# parse_args normally is used has not been tested or even intentional
###########################################
def parse_args(args):
    parser = argparse.ArgumentParser()

    #parser.add_argument("--model_config", type=str)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
#    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=True, action="store_true")
   
    parser.add_argument("-train_type", "-tt", type=str, default="lora", help="lora, full, linear, adapter")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args

args = parse_args(None)

###########################################
# Replace SGD with GaLore optimizer
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
galore_params = []
###########################################
target_modules_list = ["attn", "mlp"]
for module_name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue

    if not any(target_key in module_name for target_key in target_modules_list):
        continue
            
    print('enable GaLore for weights in module: ', module_name)
    galore_params.append(module.weight)
id_galore_params = [id(p) for p in galore_params]
# make parameters without "rank" to another group
regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
# then call galore_adamw
param_groups = [{'params': regular_params}, 
                {'params': galore_params, 'rank': 512, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params/2**20:.3f}M")
num_params = sum(p.numel() for p in model.parameters())
print(f"total parameters: {num_params/2**20:.3f}M")

optimizer = GaLoreAdamW(param_groups, lr=lr)
#optimizer = optim.Adam(model.parameters(), lr=0.01)
###########################################


# Initialize the SummaryWriter
writer = SummaryWriter('runs/experiment_name')

### Train
for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        targets = targets.cuda()

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs.last_hidden_state, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Record loss to TensorBoard - record every batch or less frequently as needed
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

# Close the writer when done to free up resources
writer.close()

split = 'test'

model.eval()
y_true = torch.tensor([])
y_score = torch.tensor([])

data_loader = train_loader_at_eval if split == 'train' else test_loader

from sklearn.metrics import accuracy_score, roc_curve, auc

with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        logits = outputs
        outputs_softmax = torch.softmax(logits, dim=-1)
        #outputs = outputs.softmax(dim=-1)
        y_true  = torch.cat((y_true , targets), 0)
        y_score = torch.cat((y_score, outputs_softmax.cpu()), 0)

    y_true = y_true.detach().numpy()
    y_score = y_score.detach().numpy()
   
    # Use argmax to find the indices of the maximum values in each row
    y_score = np.argmax(y_score, axis=1)

    # Convert to a column vector
    y_score = y_score.reshape(-1, 1)

    evaluator = Evaluator(data_flag, split, size=224)
    print(y_true)
    print(y_score)
    #metrics = evaluator.evaluate(y_score)

    accuracy = accuracy_score(y_true, y_score)
    print("rank: ", rank, "acc: ", accuracy)
    #fpr, tpr, thresholds = roc_curve(y_true, y_score)
    #auc = auc(fpr, tpr)

    #print("acc: ", accuracy, "auc: ", auc)

    #print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))
