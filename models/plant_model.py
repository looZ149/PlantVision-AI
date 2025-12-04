import torch
import torch.nn as nn
import torchvision.models as models

class PlantClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PlantClassifier, self).__init__()
        
        # input_size = channels + height + width
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):        
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)        
        return out
    

def create_plant_classifier(
        num_classes: int,
        image_channels: int = 3,
        image_height: int = 64,
        image_width: int = 64
):
    input_size = image_channels * image_height * image_width
    model = PlantClassifier(input_size, num_classes)
    return model

def create_resnet_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Erstellt ein ResNet50-Modell für die Pflanzenklassifikation.

    - Lädt ein vortrainiertes ResNet50 (ImageNet)
    - Ersetzt den letzten Fully-Connected-Layer durch einen neuen für num_classes
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # for param in model.parameters():
    #     param.requires_grad = False

    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model    


def save_model(model: nn.Module, path: str) -> None:
    """Saves only the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), path)


def load_model(
        path: str,
        num_classes: int,
        use_resnet: bool = True,
        image_channels: int = 3,
        image_height: int = 64,
        image_width: int = 64
) -> nn.Module:
    """if use_resnet=true-> loads resnet model else loads plant classifier model"""
    if use_resnet:
        model = create_resnet_model(num_classes=num_classes, pretrained=False)
    else:
        model = create_plant_classifier(
            num_classes=num_classes,
            image_channels=image_channels,
            image_height=image_height,
            image_width=image_width,
        )
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    # model.eval()  # Set the model to evaluation mode
    return model
    
    