"""
dataset.py — Dosya 2/7

Görevi: MVTec klasör yapısını (train/good, test/<defect>, ground_truth/<defect>)
okuyup PyTorch'un anlayacağı formata çevirmek.

Ana mantık: İKİ ayrı veri seti sınıfı var, bilerek ayırdık:
- Train: SADECE sağlam görüntüler (etiket yok, hepsi zaten "sağlam")
- Test: sağlam + hatalı görüntüler + hatanın TAM YERİNİ gösteren maske

Bu ayrım kodda da "biz supervised değil, sadece normal örnekten öğreniyoruz"
mantığını görünür kılıyor.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


def get_transform():
    """Görüntüyü pretrained CNN'in beklediği formata çevirir.
    KRİTİK karar: ImageNet ortalama/std ile normalize ediyoruz — bu adım
    atlanırsa pretrained ağırlıklar 'tanımadığı' bir veri dağılımı görür."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_mask_transform():
    """Maskeler için normalize YOK — 0/1 anlamı korunmalı."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


class MVTecTrainDataset(Dataset):
    """train/good içindeki sağlam görüntüleri yükler. Memory bank'i bunlardan kuracağız."""

    def __init__(self, category_root):
        self.paths = sorted(glob.glob(os.path.join(category_root, "train", "good", "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"'{category_root}/train/good/' bos veya yok.")
        self.transform = get_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class MVTecTestDataset(Dataset):
    """test/ altındaki TÜM alt klasörleri (good + her defekt türü) tarar.
    Her örnek için: görüntü, etiket (0=good, 1=defect), piksel maskesi, defekt türü."""

    def __init__(self, category_root):
        self.transform = get_transform()
        self.mask_transform = get_mask_transform()
        self.samples = []   # (img_path, label, mask_path_or_None, defect_type)

        test_root = os.path.join(category_root, "test")
        for defect_type in sorted(os.listdir(test_root)):
            for img_path in sorted(glob.glob(os.path.join(test_root, defect_type, "*.png"))):
                if defect_type == "good":
                    self.samples.append((img_path, 0, None, "good"))
                else:
                    fname = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(category_root, "ground_truth", defect_type, f"{fname}_mask.png")
                    self.samples.append((img_path, 1, mask_path, defect_type))

        if not self.samples:
            raise FileNotFoundError(f"'{test_root}' icinde ornek bulunamadi.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path, defect_type = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))

        if mask_path and os.path.exists(mask_path):
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, config.IMAGE_SIZE, config.IMAGE_SIZE)

        return {"image": img, "label": label, "mask": mask,
                "defect_type": defect_type, "path": img_path}


if __name__ == "__main__":
    # Hizli dogrulama - elimizdeki sentetik veriyle test ediyoruz
    root = "/home/claude/data/bottle_lowjitter"
    train_ds = MVTecTrainDataset(root)
    test_ds = MVTecTestDataset(root)

    print(f"Train ornek sayisi: {len(train_ds)}")
    print(f"Test ornek sayisi : {len(test_ds)}")
    print(f"Train goruntu shape: {train_ds[0].shape}")

    sample = test_ds[15]
    print(f"Test ornegi -> label: {sample['label']}, defekt: {sample['defect_type']}")