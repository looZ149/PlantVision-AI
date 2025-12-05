# PlantVision AI – Bilderkennung für Pflanzenarten 🌱

## Projektbeschreibung

PlantVision AI ist eine Anwendung zur automatischen Erkennung von Pflanzenarten anhand von Bildern.
Ein Nutzer lädt ein Foto einer Pflanze hoch, und unser Modell gibt die wahrscheinlichste Art aus, inkl. einer Konfidenzangabe.

## Ziele

- Aufbau eines Deep-Learning-Modells zur Klassifikation von Pflanzenarten
- Bereitstellung eines REST-Backends zur Bildanalyse
- Entwicklung eines einfachen Frontends zur Interaktion für Endnutzer
- Saubere Teamarbeit mit klaren Modulen, Git-Workflow und Dokumentation

---

## Tech-Stack

- **Programmiersprache:** Python
- **ML / Deep Learning:** PyTorch, Torchvision
- **Bildverarbeitung:** Pillow (optional OpenCV)
- **Backend:** FastAPI oder Flask (REST-API)
- **Frontend:** einfache Weboberfläche (HTML/CSS/JS oder React)
- **Sonstiges:** Git, virtualenv/conda, ggf. Docker

---
## RNN/Transformer für Text

## Custom Trainings-Loops (for epoch in range(...): ...)

# ToDo Data and Training

## Prio 1 – Data & Training lauffähig machen

### 1. Datenset & DataLoader (data/)

**Dateien:**
- `data/dataloader.py`
- `data/transforms.py`

**Aufgaben:**
- Oxford Flowers Dataset laden (Bilder + Labels)
- Train/Valid/Test-Splits definieren
- `torch.utils.data.Dataset` + `DataLoader` implementieren
- Standard-Transforms:
  - Resize/Crop auf konsistente Größe (z.B. 224x224)
  - `ToTensor()`
  - Normalize mit ImageNet-Mean/Std (für ResNet)

---

### 2. ResNet-Modell (models/)

**Datei:**
- `models/plant_model.py`

**Aufgaben:**
- Funktion `create_resnet_model(num_classes: int)` implementieren:
  - `torchvision.models.resnet50` mit vortrainierten Gewichten laden
  - letzten Fully-Connected-Layer (`model.fc`) an `num_classes` anpassen
- vorhandene `save_model` und `load_model` weiterverwenden:
  - `save_model(model, path)`
  - `load_model(path, num_classes, ...)` → ResNet-Variante laden

---

### 3. Trainings-Skript (training/)

**Datei:**
- `training/train.py`

**Aufgaben:**
- `train()`-Funktion implementieren:
  - Modell über `create_resnet_model(...)` holen
  - DataLoader aus `data/dataloader.py`
  - Loss: `nn.CrossEntropyLoss`
  - Optimizer: z.B. `torch.optim.Adam` oder `SGD`
  - Training über X Epochen (z.B. 5–10)
  - am Ende Gewichte speichern (`save_model(...)` → z.B. `models/Plant_classifier.pth`)
- Optional: einfache Konsolenausgabe für Loss/Accuracy pro Epoche

---

### 4. Prediction (z.B. predict.py)

**Datei:**
- `predict.py` im Projektroot oder `predict/predict.py`

**Aufgaben:**
- Modell über `load_model` laden
- Bildpfad als Argument entgegennehmen:
  ```bash
  python predict.py path/to/image.jpg

### 5. Startmenü 