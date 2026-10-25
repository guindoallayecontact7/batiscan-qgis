"""Étape 3 : entraînement d'un U-Net « bâti sahélien » (PyTorch, GPU).

python 03_entrainer.py <dossier_jeu> <dossier_sortie> --validation Doumanani [--epoques 60]

<dossier_jeu> contient un sous-dossier par village et par fond, nommé
<village>_<fond> (sortie de 02_jeu.py). Les villages listés dans --validation
ne servent jamais à l'entraînement : ils mesurent ce que le modèle fait sur
un village qu'il n'a pas vu.
"""

import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MOYENNE = np.array([0.485, 0.456, 0.406], np.float32)
ECART = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255


def lister(dossier, villages):
    paires = []
    for d in sorted(glob.glob(os.path.join(dossier, "*"))):
        village = os.path.basename(d).rsplit("_", 1)[0]
        if village not in villages:
            continue
        for img in sorted(glob.glob(os.path.join(d, "img_*.png"))):
            paires.append((img, img.replace("img_", "msk_")))
    return paires


class Vignettes(Dataset):
    def __init__(self, paires, taille=384, augmenter=True):
        self.paires = paires
        self.taille = taille
        self.augmenter = augmenter

    def __len__(self):
        return len(self.paires)

    def __getitem__(self, i):
        img = cv2.imread(self.paires[i][0])[..., ::-1]
        msk = cv2.imread(self.paires[i][1], cv2.IMREAD_GRAYSCALE)
        if self.augmenter:
            img, msk = self._augmenter(img, msk)
        x = (img.astype(np.float32) / 255.0 - MOYENNE) / ECART
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(msk.astype(np.int64))

    def _augmenter(self, img, msk):
        # échelle : les fonds n'ont pas tous la même résolution réelle
        s = random.uniform(0.8, 1.25)
        if abs(s - 1) > 0.02:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        h, w = msk.shape
        t = self.taille
        if h < t or w < t:
            img = cv2.copyMakeBorder(img, 0, max(0, t - h), 0, max(0, t - w), cv2.BORDER_REFLECT)
            msk = cv2.copyMakeBorder(msk, 0, max(0, t - h), 0, max(0, t - w), cv2.BORDER_CONSTANT, value=IGNORE)
            h, w = msk.shape
        r, c = random.randint(0, h - t), random.randint(0, w - t)
        img, msk = img[r:r + t, c:c + t], msk[r:r + t, c:c + t]
        k = random.randint(0, 3)
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        img = np.ascontiguousarray(img).astype(np.float32)
        # couleurs : saison, capteur, compression varient d'un fond à l'autre
        img = img * random.uniform(0.75, 1.25) + random.uniform(-25, 25)
        img = img * np.random.uniform(0.9, 1.1, 3)[None, None, :]
        gris = img.mean(axis=2, keepdims=True)
        img = gris + (img - gris) * random.uniform(0.6, 1.3)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
        return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(msk)


def perte(logits, cible, poids):
    ce = F.cross_entropy(logits, cible, weight=poids, ignore_index=IGNORE)
    # Dice sur « bâtiment » (intérieur + contour)
    p = logits.softmax(1)[:, 1:].sum(1)
    valide = cible != IGNORE
    g = ((cible == 1) | (cible == 2)).float()
    inter = (p * g * valide).sum()
    dice = 1 - (2 * inter + 1) / ((p * valide).sum() + (g * valide).sum() + 1)
    return ce + dice


@torch.no_grad()
def evaluer(modele, chargeur, dev):
    modele.eval()
    inter = union = tp = fp = fn = 0
    for x, y in chargeur:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = modele(x).argmax(1)
        valide = y != IGNORE
        pb = (pred > 0) & valide
        gb = (y > 0) & (y != IGNORE)
        inter += (pb & gb).sum().item()
        union += (pb | gb).sum().item()
        tp += (pb & gb).sum().item()
        fp += (pb & ~gb).sum().item()
        fn += (~pb & gb & valide).sum().item()
    modele.train()
    prec = tp / max(tp + fp, 1)
    rapp = tp / max(tp + fn, 1)
    return {"iou": inter / max(union, 1), "precision": prec, "rappel": rapp,
            "f1": 2 * prec * rapp / max(prec + rapp, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jeu")
    ap.add_argument("sortie")
    ap.add_argument("--validation", nargs="+", required=True)
    ap.add_argument("--epoques", type=int, default=60)
    ap.add_argument("--lot", type=int, default=12)
    ap.add_argument("--encodeur", default="resnet34")
    ap.add_argument("--taille", type=int, default=384)
    a = ap.parse_args()

    tous = {os.path.basename(d).rsplit("_", 1)[0] for d in glob.glob(os.path.join(a.jeu, "*"))}
    entr = lister(a.jeu, tous - set(a.validation))
    val = lister(a.jeu, set(a.validation))
    print(f"entraînement : {len(entr)} vignettes ({sorted(tous - set(a.validation))}) ; "
          f"validation : {len(val)} ({a.validation})", flush=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    ch_e = DataLoader(Vignettes(entr, a.taille), batch_size=a.lot, shuffle=True, num_workers=4,
                      drop_last=True, persistent_workers=True)
    ch_v = DataLoader(Vignettes(val, augmenter=False), batch_size=4, num_workers=2)
    modele = smp.Unet(a.encodeur, encoder_weights="imagenet", classes=3).to(dev)
    poids = torch.tensor([1.0, 1.0, 3.0], device=dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epoques * len(ch_e))
    scaler = torch.amp.GradScaler()
    os.makedirs(a.sortie, exist_ok=True)
    meilleur, journal = -1, []
    for ep in range(1, a.epoques + 1):
        t0, cumul = time.time(), 0.0
        for x, y in ch_e:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                l = perte(modele(x).float(), y, poids)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            cumul += l.item()
        m = evaluer(modele, ch_v, dev)
        m.update(epoque=ep, perte=cumul / len(ch_e), duree_s=round(time.time() - t0, 1))
        journal.append(m)
        print(json.dumps(m), flush=True)
        if m["f1"] > meilleur:
            meilleur = m["f1"]
            torch.save({"etat": modele.state_dict(), "encodeur": a.encodeur, "metriques": m},
                       os.path.join(a.sortie, "meilleur.pt"))
    with open(os.path.join(a.sortie, "journal.json"), "w") as f:
        json.dump(journal, f, indent=1)


if __name__ == "__main__":
    main()
