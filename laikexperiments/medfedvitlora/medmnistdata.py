from medmnist import PathMNIST
dataset = PathMNIST(split="train", download=True, size=224)

print(dataset[0])
