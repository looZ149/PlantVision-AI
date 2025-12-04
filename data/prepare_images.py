from pathlib import Path
import torch
from pillow import load_image, make_eval_pipeline, to_numpy, iter_images

#Pipeline erzeugen (wie torchvision transforms)
pipeline = make_eval_pipeline(
    size=224,         
    shorter=256,
    strategy="center" 
)

#jeder Bildpfad im img-Ordner durchgehen
root = Path("img")
output = Path("img_to_Torch") # Ausgabeordner
output.mkdir(exist_ok=True)

for image_path in iter_images(root):
    print("Bearbeite:", image_path)

#Bild laden
    img = load_image(image_path)

#Transformation anwenden
    img_transformed = pipeline(img)

#In numpy → Tensor
    arr = to_numpy(img_transformed)
    tensor = torch.tensor(arr)  # shape: (3, 224, 224)

#speichern als .pt Datei
    save_path = output / (image_path.stem + ".pt")
    torch.save(tensor, save_path)

print("Fertig!")
