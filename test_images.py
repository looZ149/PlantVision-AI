from pathlib import Path
from data.pillow import (
    load_image,
    resize_shorter_side,
    center_crop,
    letterbox_square,
    make_eval_pipeline,
    to_numpy,
    iter_images,
)

DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)


def process_image(path: Path):
    print(f"\n=== Verarbeite: {path} ===")

    # Bild laden
    img = load_image(path)
    print("Originalgröße:", img.size)

    # 1. Resize shorter side
    resized = resize_shorter_side(img, 256)
    resized.save(DEBUG_DIR / f"{path.stem}_resized.jpg")

    # 2. Center Crop
    center = center_crop(img, 224)
    center.save(DEBUG_DIR / f"{path.stem}_center.jpg")

    # 3. Letterbox Square
    letter = letterbox_square(img, 224)
    letter.save(DEBUG_DIR / f"{path.stem}_letterbox.jpg")

    # 4. Pipeline
    pipe = make_eval_pipeline(size=224, shorter=256, strategy="center")
    piped = pipe(img)
    piped.save(DEBUG_DIR / f"{path.stem}_pipeline.jpg")

    # 5. NumPy Konvertierung
    arr = to_numpy(piped)
    print("CHW NumPy Shape:", arr.shape)
    print("Min/Max nach Normalize:", arr.min(), "/", arr.max())


def main():
    img_root = Path("img")

    print("Suche nach Bildern in:", img_root.absolute())

    # iter_images findet rekursiv Bilder wie /Efeu/Efeu01.jpg usw.
    for img_path in iter_images(img_root):
        process_image(img_path)

    print("\nFertig! Ergebnisse liegen im Ordner /debug\n")


if __name__ == "__main__":
    main()
