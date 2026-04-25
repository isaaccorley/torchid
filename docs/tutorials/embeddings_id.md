# Intrinsic dimension of pretrained image embeddings

How does intrinsic dimension differ across pretrained vision encoders? This notebook samples a small batch of images, runs them through a few `timm` models, and estimates the intrinsic dimension of the resulting embedding clouds with several `torchid` estimators.

Setup:

```bash
uv sync --group docs
```

```python
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision
from torchvision import transforms

from torchid.estimators import MLE, lPCA, MADA, TwoNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
device
```

## Get a batch of images

We use CIFAR-10 here because it downloads in seconds and is enough to show meaningful structure. The estimators don't care that the images are tiny — they only see the embedding output.

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
ds = torchvision.datasets.CIFAR10("/tmp/cifar", train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)

N = 1024
imgs = []
for x, _ in loader:
    imgs.append(x)
    if sum(b.shape[0] for b in imgs) >= N:
        break
imgs = torch.cat(imgs)[:N]
imgs.shape
```

## Extract embeddings from several pretrained models

We pick a small mix: a supervised CNN, a supervised ViT, and a self-supervised DINOv2. The expectation from the literature (Pope et al. 2021, Doimo et al. 2024) is that self-supervised representations have **higher** intrinsic dimension than purely supervised ones, because supervised training tends to collapse features along class-relevant directions.

```python
MODELS = [
    "resnet50.a1_in1k",
    "vit_small_patch16_224.augreg_in1k",
    "vit_small_patch14_dinov2.lvd142m",
]

@torch.inference_mode()
def embed(model_name: str, x: torch.Tensor) -> torch.Tensor:
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    out = []
    for chunk in x.split(64):
        out.append(model(chunk.to(device)).cpu())
    return torch.cat(out)

embeddings = {name: embed(name, imgs) for name in MODELS}
{k: v.shape for k, v in embeddings.items()}
```

## Estimate intrinsic dimension with four methods

Each method makes different assumptions, so the absolute numbers don't always agree — but the *ranking* across models is what we care about.

```python
ESTIMATORS = {
    "lPCA (FO)": lPCA(ver="FO"),
    "TwoNN":     TwoNN(),
    "MLE":       MLE(),
    "MADA":      MADA(),
}

results = {}
for model_name, X in embeddings.items():
    Xd = X.to(device)
    results[model_name] = {ename: type(est)().fit(Xd).dimension_ for ename, est in ESTIMATORS.items()}

import pandas as pd
df = pd.DataFrame(results).T
df
```

## Plot

```python
ax = df.plot(kind="bar", figsize=(9, 4), edgecolor="black", width=0.8)
ax.set_ylabel("intrinsic dimension")
ax.set_xlabel("")
ax.set_title(f"ID of CIFAR-10 embeddings (N={N})")
ax.legend(title="estimator", loc="upper left", bbox_to_anchor=(1.02, 1))
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
```

## Streaming the same metric across batches

If you want to monitor ID *during* training rather than as a one-shot fit, use the `IntrinsicDimension` torchmetrics adapter. It buffers features across `update()` calls and runs the estimator once on `compute()`.

```python
from torchid.metrics import IntrinsicDimension

metric = IntrinsicDimension(method="twonn", max_samples=2000).to(device)

model = timm.create_model(MODELS[0], pretrained=True, num_classes=0).to(device).eval()
with torch.inference_mode():
    for x, _ in loader:
        metric.update(model(x.to(device)))
        if metric.features and sum(b.shape[0] for b in metric.features) >= 2000:
            break

print(f"streaming TwoNN ID: {metric.compute().item():.2f}")
```

## Takeaways

- The streaming `IntrinsicDimension` metric makes it cheap to log ID per epoch during training without writing a custom buffer.
- DINOv2 typically lands at the highest ID, supervised ViT in the middle, supervised ResNet at the bottom — consistent with the literature.
- All four estimators agree on the *ranking* even when the absolute scales differ. If you only care about relative comparisons, the choice of estimator usually doesn't matter; pick the cheapest (lPCA) for live monitoring.
