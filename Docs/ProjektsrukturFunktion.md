# 🌿 PlantVision AI

**PlantVision AI** ist ein Deep-Learning-basiertes System zur Erkennung von Pflanzenarten anhand von Bildern.  
Das Modell verwendet **PyTorch** und **Transfer Learning** mit einem vortrainierten **ResNet50**, um das **Oxford Flowers 102 Dataset** zu klassifizieren.

Die Anwendung unterstützt:

- Training eines Modells
- Automatischen Download des Datasets
- Speicherung und Laden des Modells
- Klassifikation beliebiger JPG/PNG-Bilder
- Ausgabe von **Klassenname + Confidence**

---

## 📦 Projektstruktur

    ImageAI/
    │
    ├── data/
    │   ├── __init__.py
    │   └── torchvision.py       # Dataset, Transforms, DataLoader, Class Names
    │
    ├── models/
    │   ├── __init__.py
    │   ├── plant_model.py       # ResNet50 Modell + Save/Load Utilities
    │   └── flower_resnet.pth    # Gespeicherte Modelldatei (nach Training)
    │
    ├── training/
    │   ├── __init__.py
    │   └── train.py             # Training Pipeline (Transfer Learning)
    │
    ├── predict.py               # Bildklassifikation
    │
    └── README.md                # Projektbeschreibung

---

## 🧰 Anforderungen

### Python

- Python **mind. 3.10**


### Libraries installieren

**CPU-Version von PyTorch (empfohlen, einfach):**

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

**Weitere Abhängigkeiten:**

    pip install matplotlib

> Für GPU-Training ist eine passende CUDA-Version von PyTorch nötig  
> (optional, nicht notwendig für Demo).

---

## 📥 Dataset

Das Projekt verwendet:

> Oxford Flowers 102 Dataset

und lädt es **automatisch herunter**, wenn nicht vorhanden.

Kein manueller Download notwendig.

---

## 🚀 Training starten



### 1. Training starten

    python -m training.train

Das Training:

- lädt das Dataset
- erstellt ein ResNet50-Modell
- friert den Feature-Extractor ein
- trainiert den Klassifikationskopf
- validiert das Modell pro Epoche

Am Ende wird das Modell gespeichert als:

    models/flower_resnet.pth

### Standard-Parameter

In `training/train.py`:

```python
num_epochs = ....
batch_size = 32
learning_rate = 1e-3
