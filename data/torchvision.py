import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.io import decode_image



def FixMeLater():

    plt.rcParams["savefig.bbox"] = 'tight'  # Ensure tight bounding boxes for saved figures, we only need that for detection? I guess..
    # plt probably unnecessary here? atm atleast?



# We need to grab the current loaded img from pillow.py over here, to transform it.
# probably something like this?


# Dont need pillow here atm if we use the OxfordFlower dataset.
# We can take care of that once the model is trained?

# import data.pillow as img

# H, W = 256, 256  # Example height and width for the image tensor

# pil_img = img.KlassenName.VariableName # Replace with the actual variable name from pillow.py, i guess?

# Define transform ONLY for the TRAINING!
trainingTransforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

#from here we can send the transformedImg to the model for training

# Obviously.. defining transform for validation here
validationTransforms = v2.Compose([
    v2.Resize((256, 256)), 
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])



# Need another Transform for the Training with the Oxford Flowers102 dataset
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
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validationFlowersTransforms = v2.Compose([
    v2.Resize((256, 256)),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Could add an additional 3rd transformsSet for test dataset. But most likely we can use validationTransforms? Has no augmentation, so should work.. probably?

# In case we need Detection
# from torchvision import tv_tensors
# img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)  # Example image tensor
# boxes = torch.randint(0, H // 2, size=(3, 4))  # Example bounding boxes we could combine it with plt from above
# boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1 and y2 > y1
# boxes = tv_tensors.BoundBoxes(boxes, format="XYXY", image_size=(H, W)) # H and W need to be defined somewhere
# img, boxes = trainingTransforms(img, boxes)  # Apply some transforms
# output_dict = trainingTransforms({"image": img, "boxes": boxes})  # Apply same transforms to image and boxes

# Guess we can just run the file from main, and let it work. Dont necessarily need to define/call anything here?


# Load the flowerDataSet 

trainDataset = datasets.Flowers102(
    root="dataset",
    split="train",
    download=True, # Gotta dowload it once atleast. FIX ME, PUT ME INTO A FUNCTION WHICH CHECKS IF THE DATASET ALREADY EXISTS
    transform=trainingFlowersTransforms
)


validationDataset = datasets.Flowers102(
    root="dataset",
    split="val",
    download=True, # Gotta dowload it once atleast. FIX ME, PUT ME INTO A FUNCTION WHICH CHECKS IF THE DATASET ALREADY EXISTS
    transform=validationFlowersTransforms
)


testDataset = datasets.Flowers102(
    root="dataset",
    split="test",
    download=True, # Gotta dowload it once atleast. FIX ME, PUT ME INTO A FUNCTION WHICH CHECKS IF THE DATASET ALREADY EXISTS
    transform=trainingFlowersTransforms
)

# Dataloaders

def get_dataloaders(batch_size: int = 32):
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationDataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


