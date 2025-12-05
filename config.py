from data.torchvision import get_dataloaders
from models.plant_model import load_model
import torch

NUM_CLASSES = 102
#Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS  = 5
SEED = 102

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/plant_classifier.pth"
MODEL = load_model(
            path=MODEL_PATH,
            num_classes=NUM_CLASSES,
            use_resnet=True
        )
MODEL.to(DEVICE)

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)