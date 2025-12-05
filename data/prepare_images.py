from pathlib import Path
import torch
from data.pillow import load_image, make_eval_pipeline, to_numpy, iter_images
## if we just do from pillow import instead data.pillow we can get a naming conflict (apparently thats the issue?) should be fixed this way on any

Red = "\033[31m"
Green = "\033[32m"
Yellow = "\033[33m"
RESET = "\033[0m"

#Pipeline erzeugen (wie torchvision transforms)
#Happens right on import by main.py
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
    print(f"{Red}Bearbeite:{RESET}", f"{Yellow}{image_path}{RESET}")

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

print(f"{Green}Fertig!{RESET}")

print(f"{Green}Alle Bilder wurden im Ordner img_to_Torch als .pt Dateien gespeichert.{RESET}")
