"""
config.py — Dosya 1/7

Görevi: Projedeki TÜM ayarları tek yerde toplamak.

Neden ayrı bir dosya? Çünkü kod içinde gezip sayı değiştirmek yerine,
deney yaparken (örn. başka bir kategori, başka bir backbone denerken)
SADECE bu dosyaya bakman yeterli olacak. Bu ayrım, "configuration" ile
"logic"i birbirinden ayırmak diye bilinir — profesyonel projelerde standarttır.
"""

import torch

# ---- Veri ----
DATA_ROOT = "./data"      # içinde <CATEGORY>/train/good, test/... olmalı
CATEGORY = "bottle"

# ---- Görüntü ----
IMAGE_SIZE = 224           # ImageNet-pretrained CNN'lerin standart girdi boyutu
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- Feature extraction (KARAR: yukarıdaki tabloya bak) ----
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ["layer2", "layer3"]

# ---- Memory bank (KARAR) ----
CORESET_RATIO = 0.10
K_NEAREST = 1

# ---- Donanım ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Çıktı ----
OUTPUT_DIR = "./outputs"