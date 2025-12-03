import torch

from models.plant_model import create_plant_classifier

def main():
    num_classes = 5  # Example number of plant species
    image_channels = 3
    image_height = 64
    image_width = 64
    batch_size = 4

    model = create_plant_classifier(
        num_classes=num_classes,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width
    )

    print("Modell erstellt")
    print (model)

    dummy_input = torch.randn(batch_size, image_channels, image_height, image_width)
   # Run a forward pass
    output = model(dummy_input)

    print("Output-Shape:", output.shape) 

if __name__ == "__main__":
    main()