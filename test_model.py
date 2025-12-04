import torch
from models.plant_model import (
    create_plant_classifier,
    create_resnet_model,
)


def test_mlp():
    """Testet das einfache MLP-Modell (PlantClassifier)."""
    num_classes = 5
    image_channels = 3
    image_height = 64
    image_width = 64
    batch_size = 4

    print("Starte MLP-Test...")

    model = create_plant_classifier(
        num_classes=num_classes,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width,
    )

    dummy_input = torch.randn(batch_size, image_channels, image_height, image_width)
    output = model(dummy_input)

    print("MLP Output-Shape:", output.shape)  # erwartet: torch.Size([4, 5])


def test_resnet():
    """Testet das ResNet50-Modell für Pflanzen (Oxford Flowers)."""
    num_classes = 102          # Oxford Flowers 102
    image_channels = 3         # RGB
    image_height = 224         
    image_width = 224
    batch_size = 2

    print("Starte ResNet-Test...")

    model = create_resnet_model(num_classes=num_classes)

    dummy_input = torch.randn(batch_size, image_channels, image_height, image_width)
    output = model(dummy_input)

    print("ResNet Output-Shape:", output.shape)  # erwartet: torch.Size([2, 102])


def main():
    # Nur MLP testen:
    # test_mlp()

    # Nur ResNet testen:
    test_resnet()


if __name__ == "__main__":
    main()