"""
models/memory_bank.py — Dosya 4/7

Görevi: Eğitim setindeki TÜM patch feature'larını toplayıp bir "hafıza
havuzuna" (memory bank) koymak, ardından coreset subsampling ile akıllıca
küçültmek.

Neden küçültme gerekiyor? Gerçek bir kategori için: 200 görüntü x 196 patch
= 39.200 vektör. Test sırasında her patch için bunların TÜMÜYLE mesafe
hesaplamak yavaş olur. Coreset subsampling (greedy k-center), havuzu
küçültürken çeşitliliği/temsil gücünü mümkün olduğunca korur.

Greedy K-Center mantığı:
1. Rastgele bir nokta seç.
2. Şu ana kadar seçilenlere en UZAK olan noktayı bul, ekle.
3. Tekrarla.
Böylece kümenin "şekli" (genel dağılımı) korunur — rastgele seçim yapsaydık
yoğun bölgeler fazla, seyrek bölgeler az temsil edilirdi.
"""

import torch
from torch.utils.data import DataLoader

from dataset import MVTecTrainDataset
from models.feature_extractor import PatchFeatureExtractor
import config


def greedy_coreset_subsample(features: torch.Tensor, n_select: int):
    """features: (N, C) -> n_select adet indeks döndürür."""
    N = features.shape[0]
    if n_select >= N:
        return torch.arange(N)

    selected = [0]
    min_distances = torch.cdist(features, features[0:1]).squeeze(1)

    for _ in range(1, n_select):
        next_idx = int(torch.argmax(min_distances).item())
        selected.append(next_idx)
        new_dist = torch.cdist(features, features[next_idx:next_idx + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, new_dist)

    return torch.tensor(selected)


class MemoryBank:
    def __init__(self, coreset_ratio=config.CORESET_RATIO):
        self.coreset_ratio = coreset_ratio
        self.bank = None
        self.grid_size = None

    def build(self, category_root, extractor: PatchFeatureExtractor, batch_size=8):
        train_ds = MVTecTrainDataset(category_root)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        all_vectors = []
        for batch in loader:
            patches, grid_size = extractor(batch)
            self.grid_size = grid_size
            B, N_patch, C = patches.shape
            all_vectors.append(patches.reshape(B * N_patch, C))

        raw_bank = torch.cat(all_vectors, dim=0)
        print(f"Ham havuz: {raw_bank.shape[0]} vektor ({len(train_ds)} goruntu)")

        n_select = max(1, int(raw_bank.shape[0] * self.coreset_ratio))
        print(f"Coreset ile kucultuluyor: {raw_bank.shape[0]} -> {n_select}")
        idx = greedy_coreset_subsample(raw_bank, n_select)
        self.bank = raw_bank[idx]

        print(f"Final memory bank: {self.bank.shape}")
        return self.bank

    def save(self, path):
        torch.save({"bank": self.bank, "grid_size": self.grid_size}, path)

    def load(self, path):
        data = torch.load(path)
        self.bank = data["bank"]
        self.grid_size = data["grid_size"]
        return self.bank


if __name__ == "__main__":
    extractor = PatchFeatureExtractor(pretrained=False)  # sandbox testi
    bank = MemoryBank(coreset_ratio=0.10)
    bank.build("/home/claude/data/bottle_lowjitter", extractor)
    