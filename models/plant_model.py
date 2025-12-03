import torch
import torch.nn as nn

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
    

def create_plant_classifier(num_classes: int, image_channels: int = 3, image_height: int = 64, image_width: int = 64):
    input_size = image_channels * image_height * image_width
    model = PlantClassifier(input_size, num_classes)
    return model   