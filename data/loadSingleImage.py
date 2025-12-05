from pathlib import Path
from PIL import Image
import predict as pred

Green = "\033[32m"
RESET = "\033[0m"

def print_in_columns(items, col_width=15, cols=5):
    for i, item in enumerate(items):
        print(f"{item:<{col_width}}", end="")
        if (i + 1) % cols == 0:
            print()
    print()  

def loadSingleImage():
    root = Path("img_to_Torch")
    imgDir = Path("img")

     # zeigt alle .pt datei an 
    for image_path in root.iterdir():
        if image_path.suffix == ".pt":
            pass 

    print(f"{Green}Available images:{RESET}") 

    names = [p.stem for p in root.iterdir() if p.suffix == ".pt"]
    names.sort()
    print_in_columns(names, col_width=15, cols=5)

    # User - Select the Image
    selected_image = input("Enter the filename of the image to load: ").strip()
    selected_image = selected_image + ".jpg"
    base_name = selected_image.replace(".jpg", ".pt")
  
    # Find the image in img directory
    found_path = None
    for folder in imgDir.iterdir():
        if folder.is_dir():
            candidate = folder / selected_image
            if candidate.exists():
                found_path = candidate
                break

    if not found_path:
        print(f"ERROR: Could not find {selected_image} in any folder inside {imgDir}")
        return
    
    # Load and show the image
    img = Image.open(found_path)
    img.show()

    # execute prediction.py from here with the selected image as param.
    pred.main(found_path) 
    


