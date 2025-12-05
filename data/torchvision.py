import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

plt.rcParams["savefig.bbox"] = 'tight' # only need that for object detection

# Define transform ONLY for the TRAINING!
trainingTransforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

# Define transform for validation
validationTransforms = v2.Compose([
    v2.Resize((256, 256)), 
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# Transform for Training with the Oxford Flowers102 dataset
trainingFlowersTransforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(20),
    v2.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2
    ),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validationFlowersTransforms = v2.Compose([
    v2.Resize((256, 256)),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the Flowers102 dataset
trainDataset = datasets.Flowers102(
    root="dataset",
    split="train",
    download=True, 
    transform=trainingFlowersTransforms
)

validationDataset = datasets.Flowers102(
    root="dataset",
    split="val",
    download=True, 
    transform=validationFlowersTransforms
)

testDataset = datasets.Flowers102(
    root="dataset",
    split="test",
    download=True, 
    transform=validationFlowersTransforms
)

# Dataloaders
def get_dataloaders(batch_size: int = 32):
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationDataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_flowers_names():
    return trainDataset.classes


