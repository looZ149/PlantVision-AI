import os
from sympy import N
import torch 
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from models.plant_model import create_resnet_model, save_model, load_model
from data.torchvision import get_dataloaders
from config import NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED, MODEL_PATH

def set_seed(seed: int) -> None:
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

criterion = nn.CrossEntropyLoss()
# num_epochs = 1 

def train(
        num_classes: int = NUM_CLASSES,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        model_out_path: str = MODEL_PATH,
        resume: bool = True,
):
    """
    train the ResNet model on the Flowers102 dataset and save the weights to a file.
    
    """
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Create model and load model if resume is True
    if resume and os.path.exists(model_out_path):
        print(f"Saved model found. Resume training from {model_out_path}")
        model = load_model(
            path=model_out_path,
            num_classes=num_classes,
            use_resnet=True
        )
        model.to(device)
        model.train()
    else:
        print("no saved model found. Start training from scratch.")
        model = create_resnet_model(num_classes=num_classes, pretrained=True)
        model.to(device)

    
    

    #optimize trainable parameters only
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Training loop
    for epoch in range(1, NUM_EPOCHS +1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n=== Epoche {epoch}/{NUM_EPOCHS} ===")

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val  Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    # Save the trained model
    save_model(model, model_out_path)
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {model_out_path}")


def evaluate(model: nn.Module, dataloader, criterion, device: torch.device):
    """
    Führt eine Evaluierung auf dem Validierungsdatensatz durch.
    Gibt (loss, accuracy) zurück.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    avg_loss = running_loss / total if total > 0 else 0
    accuracy = correct / total * 100 if total > 0 else 0
    return avg_loss, accuracy        

                                      
if __name__ == "__main__":
    train()            