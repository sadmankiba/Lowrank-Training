from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# Initialize the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10)
model = model.to('cuda')

# Prepare data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    lambda x: x.repeat(3, 1, 1)  # Repeat image channels if necessary (e.g., for grayscale images)
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
model.train()
for inputs, targets in train_loader:
    # Convert images to feature vectors
    inputs = feature_extractor(images=inputs, return_tensors="pt")['pixel_values'].to('cuda')
    targets = targets.to('cuda')

    optimizer.zero_grad()
    outputs = model(inputs)
    logits = outputs.logits  # This will correctly reference logits
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

