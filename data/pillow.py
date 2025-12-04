
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Resampling

__all__ = [
    "load_image",
    "ensure_rgb",
    "from_ndarray_bgr",
    "save_jpeg",
    "resize_shorter_side",
    "center_crop",
    "letterbox_square",
    "make_eval_pipeline",
    "to_numpy",
    "iter_images",
]

###### ---- Bild IO ---- ######
 ## Lädt ein Bild, korrigiert EXIF-Orientierung und erzwingt RGB.
 ## Warum: Handy-Fotos haben Rotations-EXIF; Modelle erwarten 3 Kanäle.

def load_image(path: str | Path, exif_transpose: bool = True, to_rgb: bool = True) -> Image.Image:
    p = Path(path)
    with Image.open(p) as im:
        im.load()
        if exif_transpose:
            im = ImageOps.exif_transpose(im)
        if to_rgb and im.mode != "RGB":
            im = im.convert("RGB")
        return im

## Erzwingt RGB-Modus; verwirft Alpha-Kanal.
def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

## Konvertiert ein HxWx3 BGR-NumPy-Array in ein PIL RGB-Bild.
def from_ndarray_bgr(arr: np.ndarray) -> Image.Image:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("erwarte HxWx3 BGR-Array")
    return Image.fromarray(arr[..., ::-1], mode="RGB")

## Speichert ein JPEG mit definierten Parametern.
def save_jpeg(
    img: Image.Image,
    path: str | Path,
    quality: int = 92,
    progressive: bool = True,
    optimize: bool = True,
    strip_exif: bool = True,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = img
    if strip_exif:
        # Kopie ohne Metadaten – reduziert Inkonsistenzen
        tmp = Image.new(out.mode, out.size)
        tmp.putdata(list(out.getdata()))
        out = tmp
    out.save(
        p,
        format="JPEG",
        quality=int(quality),
        progressive=bool(progressive),
        optimize=bool(optimize),
        subsampling="4:2:0",
    )


###### ---- Bild Geometrie ---- ######
## Resizing & Cropping Funktionen, die in Pipelines verwendet werden.
def resize_shorter_side(img: Image.Image, size: int, resample: Resampling = Resampling.BICUBIC) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("ungültige Bildgröße")
    if w < h:
        new_w, new_h = size, max(1, int(round(h * size / w)))
    else:
        new_w, new_h = max(1, int(round(w * size / h))), size
    return img.resize((new_w, new_h), resample=resample)

## Deterministischer Center-Crop.
def center_crop(img: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(img, (size, size), method=Resampling.BICUBIC, centering=(0.5, 0.5))

## Letterbox-Crop mit Padding, um ein quadratisches Bild zu erzeugen.
def letterbox_square(img: Image.Image, size: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Resampling.BICUBIC)
    out = Image.new("RGB", (size, size), fill)
    out.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return out


###### ---- Pipeline & Utils ---- ######
## Erzeugt eine Eval-Pipeline basierend auf der gewählten Strategie, Größe und kürzerer Seite.
def make_eval_pipeline(
    size: int = 224,
    shorter: int = 256,
    strategy: Literal["center", "letterbox", "resize_shorter"] = "center",
) -> Callable[[Image.Image], Image.Image]:
    def _pipe(img: Image.Image) -> Image.Image:
        if strategy == "center":
            return center_crop(resize_shorter_side(img, max(shorter, size)), size)
        if strategy == "letterbox":
            return letterbox_square(img, size)
        if strategy == "resize_shorter":
            return center_crop(resize_shorter_side(img, max(shorter, size)), size)
        raise ValueError(f"unbekannte Strategie: {strategy}")
    return _pipe

## Konvertiert ein PIL-Bild in ein NumPy-Array mit optionaler Normalisierung.
def to_numpy(
    img: Image.Image,
    chw: bool = True,
    normalize: Literal["imagenet", "none"] = "imagenet",
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    if chw:
        arr = arr.transpose(2, 0, 1)
    return arr

## Iteriert über Bilddateien in einem Verzeichnis, basierend auf Erweiterungen.
def iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

######
######
