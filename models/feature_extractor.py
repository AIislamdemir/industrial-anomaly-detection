"""
models/feature_extractor.py — Dosya 3/7

Görevi: Görüntüyü pretrained bir CNN'den (wide_resnet50_2) geçirip,
ara katmanlardan (layer2, layer3) patch-level öznitelik (feature) çıkarmak.

Ana mantık: PyTorch modelleri varsayılan olarak sadece SON çıktıyı verir.
Ara katmanlara ulaşmak için "forward hook" kullanıyoruz — bu, veri o
katmandan geçtiği anda bizim bir fonksiyonumuzu tetikleyip o veriyi
yakalamamızı sağlıyor.

Neden layer2+layer3? (config.py'da karar verdik)
- Erken katmanlar: çok genel (sadece kenar/renk), ayırt edici değil
- layer2+layer3: doku/desen bilgisi + hâlâ KONUM bilgisi taşıyor (lokalizasyon için şart)
- layer4: çok soyut/sınıf-özel, konum bilgisi büyük ölçüde kaybolur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class PatchFeatureExtractor(nn.Module):
    def __init__(self, backbone_name=config.BACKBONE, layers=config.FEATURE_LAYERS,
                 pretrained=True, local_pool_size=3):
        super().__init__()
        self.layers = layers
        self.local_pool_size = local_pool_size

        # NOT: pretrained=True ImageNet agirliklarini internetten indirir.
        # Bu sandbox'ta o adrese erisim kapali - test icin pretrained=False kullanacagiz.
        # Kendi makinende pretrained=True ile calistir, gercek kalite oradan gelir.
        weights = None
        if pretrained:
            weights_map = {
                "wide_resnet50_2": models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
                "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
            }
            weights = weights_map.get(backbone_name)

        backbone_fn = getattr(models, backbone_name)
        self.backbone = backbone_fn(weights=weights)
        self.backbone.eval()

        # Agirliklari DONDURMUYORUZ - PatchCore hic egitim/fine-tuning gerektirmez,
        # pretrained feature'lari "oldugu gibi" kullanir.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Hook'larla ara katman ciktilarini yakala
        self._features = {}
        for layer_name in self.layers:
            layer = dict(self.backbone.named_modules())[layer_name]
            layer.register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x):
        """
        Girdi : (B, 3, H, W) görüntü batch'i
        Çıktı : (B, N_patch, C) birleştirilmiş patch-feature'lar + (n_h, n_w) grid boyutu
        """
        self._features = {}
        _ = self.backbone(x)

        # layer2 ve layer3'ün çözünürlükleri farklı (örn. 28x28 vs 14x14).
        # Kanal-bazlı birleştirmek için EN KÜÇÜK çözünürlüğe (en derin katman) hizalıyoruz.
        target_size = self._features[self.layers[-1]].shape[-2:]

        aligned = []
        for name in self.layers:
            fmap = self._features[name]
            if self.local_pool_size > 1:
                # Yerel komsuluk ortalaması (PatchCore'un "locally aware" adımı)
                fmap = F.avg_pool2d(fmap, kernel_size=self.local_pool_size,
                                     stride=1, padding=self.local_pool_size // 2)
            if fmap.shape[-2:] != target_size:
                fmap = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(fmap)

        combined = torch.cat(aligned, dim=1)            # (B, C_total, n_h, n_w)
        B, C, n_h, n_w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, n_h * n_w, C)
        return patches, (n_h, n_w)


if __name__ == "__main__":
    # Shape/mimari testi - gerçek kalite için kendi makinende pretrained=True kullan
    extractor = PatchFeatureExtractor(pretrained=False)
    dummy = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    patches, (n_h, n_w) = extractor(dummy)

    print(f"Backbone: {config.BACKBONE}, katmanlar: {config.FEATURE_LAYERS}")
    print(f"Patch grid: {n_h}x{n_w} = {n_h*n_w} patch/goruntu")
    print(f"Cikti shape: {patches.shape}  (batch, n_patch, feature_dim)")