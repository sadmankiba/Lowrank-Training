from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'pathmnist'
download = True

NUM_EPOCHS = 20
BATCH_SIZE = 128
lr = 0.001

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
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

info = INFO[data_flag]
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224, mmap_mode='r')

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

x, y = train_dataset[0]

print(x.shape, y.shape)


####################################################
### Get foundation model

from torchvision.models import resnet18

#model = resnet18(num_classes=n_classes).cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRAAdapter, self).__init__()
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, output_dim)

    def forward(self, x):
        x_down = self.down(x)
        x_act = torch.relu(x_down)  # Optional: Add non-linearity
        x_up = self.up(x_act)
        return x + x_up  # Skip connection to facilitate learning

from torchvision.models import resnet18
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes, rank=64):
        super(ModifiedResNet18, self).__init__()
        # Load a pre-trained ResNet
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer

        # Low rank adapter
        self.lora = LoRAAdapter(input_dim=512, output_dim=512, rank=rank)  # Assuming the adapter is for the last layer

        # Final classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.lora(x)
        x = self.fc(x)
        return x


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# Instantiate the model
model = ModifiedResNet18(num_classes=n_classes, rank=64).cuda()

# Freeze the parameters of the base model
freeze_parameters(model.base_model)

# The LoRA adapter and the classifier should remain trainable


optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
# train

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        targets = targets.cuda()

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


split = 'test'

model.eval()
y_true = torch.tensor([])
y_score = torch.tensor([])

data_loader = train_loader_at_eval if split == 'train' else test_loader

with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = outputs.softmax(dim=-1)
        y_score = torch.cat((y_score, outputs.cpu()), 0)

    y_score = y_score.detach().numpy()

    evaluator = Evaluator(data_flag, split, size=224)
    metrics = evaluator.evaluate(y_score)

    print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))


