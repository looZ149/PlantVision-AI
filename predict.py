import os
import torch
import sys

from torchvision.io import read_image
from pathlib import Path
from models.plant_model import create_resnet_model
from data.torchvision import validationFlowersTransforms, get_flowers_names
from config import NUM_CLASSES, MODEL_PATH

Model_Path = "models/plant_classifier.pth"


def load_trained_model(
        weights_path: str = Model_Path,
        num_classes: int = NUM_CLASSES,
        device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    creates a restnet-model & loads trainied weights.
    
    """
    if isinstance(device, str):
        device = torch.device(device)

    model = create_resnet_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path: str | Path) -> torch.Tensor:

    #loads and preprocesses an image for prediction.
    image_path = Path(image_path)

    if not image_path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load image: Tensor [C, H, W] with dtype uint8
    img = read_image(str(image_path))

    # 3 channels expected
    if img.shape[0] ==1:
        img = img.repeat(3, 1, 1)  # convert grayscale to RGB by repeating channels
    elif img.shape[0] ==4:
        img = img[:3, :, :]  #only keep RGB channels

    img = validationFlowersTransforms(img)  # Apply validation transforms

    img = img.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

    return img


def predict_image(
        
        image_path:str,
        weights_path: str = MODEL_PATH,   #loads model and image for prediction and returns predicted class index.
        num_classes: int = NUM_CLASSES,
        

):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    print(f"Preprocessing image from {image_path}")
    img_tensor = preprocess_image(image_path).to(device)

    # image_path = Path(image_path)

    # load model
    print(f"Loading model from {weights_path}")
    model = load_trained_model(
        weights_path=weights_path,
        num_classes=num_classes,
        device=device
    )

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    
    class_names = get_flowers_names()
    if 0 <= pred_idx < len(class_names):
        class_name = class_names[pred_idx]
    else:
        class_name = f"Unbekannte Klasse (Index {pred_idx})"
 
    print("\n=== Vorhersage ===")
    print(f"Klasse: {class_name}")
    print(f"Index: {pred_idx}")
    print(f"Confidence: {confidence:.4f}")


def main(image):
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]    


if __name__ == "__main__":
    main()    



    
    
    