"""compute_latent.py — make a TacGen latent for a NEW object from close-up photo(s).

Produces a 128-d latent the SAME way object_latents.npz was built
(embed_analyze.py): DINOv2 ViT-S/14 -> proj_v (from best.pt) -> L2-normalize,
averaged over the object's photos and re-normalized. Feeding a latent made any
other way would be out-of-distribution vs training and invalidate the test.

Runs ANYWHERE that has torch + torchvision + best.pt — your laptop is easiest,
since the photos are already there (CPU is fine; the result is machine-independent
to well within cosine noise). One-time: `pip install torch torchvision pillow numpy`,
and have best.pt locally (your path_d copy, or scp it once from the server:
/export/wy891/home/_tacgen_probe_tmp/best.pt). First run downloads DINOv2 via torch.hub.

Usage:
  python compute_latent.py <best.pt> <out.npy> <photo1.jpg> [photo2.jpg ...]
Example (on your laptop):
  python compute_latent.py best.pt rubber_duck.npy duck1.jpg duck2.jpg
Then move rubber_duck.npy to Thor (USB) and run eval_client_tacin.py --latent-npy rubber_duck.npy.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


class Proj(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) * 0.02)
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        return F.normalize(F.linear(x, self.weight, self.bias), dim=-1)


tf = T.Compose([
    T.Resize(224, antialias=True), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: python compute_latent.py <best.pt> <out.npy> <photo1.jpg> [photo2.jpg ...]")
    ckpt_path, out_path, photos = sys.argv[1], sys.argv[2], sys.argv[3:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=dev)
    m = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    pv = {k.replace("proj_v.", ""): v for k, v in m.items() if k.startswith("proj_v.")}
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    proj = Proj(384, pv["weight"].shape[0]).to(dev).eval()
    proj.load_state_dict(pv)

    Z = []
    with torch.no_grad():
        for p in photos:
            z = proj(dino(tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))).squeeze(0).cpu().numpy()
            Z.append(z)
    Z = np.stack(Z)
    z = Z.mean(0)
    z /= np.linalg.norm(z) + 1e-8          # unit vector, same as training latents
    np.save(out_path, z.astype(np.float32))
    print(f"saved {out_path}  shape={z.shape}  from {len(photos)} photo(s)")
    if len(photos) > 1:
        S = Z @ Z.T
        iu = np.triu_indices(len(photos), 1)
        intra = float(S[iu].mean())
        print(f"intra-cosine across your photos = {intra:.3f}  {'OK' if intra > 0.5 else 'LOW - photos may be inconsistent'}")


if __name__ == "__main__":
    main()
