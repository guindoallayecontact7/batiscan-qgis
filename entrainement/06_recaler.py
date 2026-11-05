"""Recale les masques issus de sources ouvertes sur l'image, à partir des prédictions d'un modèle.

python 06_recaler.py <point_de_controle.pt> <jeu_source> <jeu_sortie> [--max 30]

Pour chaque vignette, on cherche la translation (±max px) qui superpose le mieux le masque
à la prédiction. Le jeu source n'est pas modifié : images liées, masques réécrits dans
<jeu_sortie>. Les vignettes sans accord suffisant sont écartées. Bilan dans recalage.csv.
"""
import argparse, csv, glob, os
import cv2, numpy as np, segmentation_models_pytorch as smp, torch

MOYENNE = np.array([0.485, 0.456, 0.406], np.float32)
ECART = np.array([0.229, 0.224, 0.225], np.float32)


def correlation(p, g, r):
    """Carte des recouvrements p·décalage(g) pour les translations de ±r px."""
    fp = np.fft.rfft2(p, s=(p.shape[0] + 2 * r, p.shape[1] + 2 * r))
    fg = np.fft.rfft2(g, s=(p.shape[0] + 2 * r, p.shape[1] + 2 * r))
    c = np.fft.irfft2(fp * np.conj(fg), s=(p.shape[0] + 2 * r, p.shape[1] + 2 * r))
    return np.fft.fftshift(c)[c.shape[0] // 2 - r:c.shape[0] // 2 + r + 1,
                              c.shape[1] // 2 - r:c.shape[1] // 2 + r + 1]


def deplacer(m, dy, dx):
    s = np.full_like(m, 255)
    h, w = m.shape
    s[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return s


def iou(p, g):
    return (p & g).sum() / max((p | g).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modele"); ap.add_argument("source"); ap.add_argument("sortie")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--seuil", type=float, default=0.15, help="IoU minimale après recalage")
    a = ap.parse_args()
    ck = torch.load(a.modele, map_location="cuda")
    net = smp.Unet(ck["encodeur"], encoder_weights=None, classes=3).cuda().eval()
    net.load_state_dict(ck["etat"])
    os.makedirs(a.sortie, exist_ok=True)
    lignes = []
    for d in sorted(glob.glob(os.path.join(a.source, "*"))):
        if not os.path.isdir(d):
            continue
        nom = os.path.basename(d); res = []
        for f in sorted(glob.glob(os.path.join(d, "img_*.png"))):
            m = cv2.imread(f.replace("img_", "msk_"), cv2.IMREAD_GRAYSCALE)
            g = (m == 1) | (m == 2)
            if g.mean() < 0.005:
                res.append((f, m, None)); continue
            x = (cv2.imread(f)[..., ::-1].astype(np.float32) / 255 - MOYENNE) / ECART
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                q = net(torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda())
            p = q[0].float().softmax(0)[1:].sum(0).cpu().numpy()
            c = correlation(p, g.astype(np.float32), a.max)
            iy, ix = np.unravel_index(c.argmax(), c.shape)
            dy, dx = int(iy - a.max), int(ix - a.max)
            pb = p > 0.5
            res.append((f, m, (dy, dx, iou(pb, g), iou(pb, deplacer(g.astype(np.uint8), dy, dx) == 1))))
        bons = [r[2] for r in res if r[2] and r[2][3] >= 0.3]
        med = (int(np.median([b[0] for b in bons])), int(np.median([b[1] for b in bons]))) if bons else None
        os.makedirs(os.path.join(a.sortie, nom), exist_ok=True); gardes = 0
        for f, m, s in res:
            if s is None:
                dy, dx = med or (0, 0); i0 = i1 = float("nan"); ok = med is not None
            else:
                dy, dx, i0, i1 = s
                # une vignette peu fiable suit le décalage médian du village
                if i1 < 0.3 and med:
                    dy, dx = med
                ok = i1 >= a.seuil
            lignes.append([nom, os.path.basename(f), dy, dx, i0, i1, int(ok)])
            if not ok:
                continue
            o = os.path.join(a.sortie, nom, os.path.basename(f))
            if not os.path.exists(o):
                os.link(f, o)
            cv2.imwrite(o.replace("img_", "msk_"), deplacer(m, dy, dx)); gardes += 1
        print(nom, "médian", med, "vignettes", gardes, "/", len(res), flush=True)
    with open(os.path.join(a.sortie, "recalage.csv"), "w", newline="") as h:
        w = csv.writer(h); w.writerow("village vignette dy dx iou_avant iou_apres garde".split()); w.writerows(lignes)
    t = [l for l in lignes if l[4] == l[4]]
    print("IoU moyen avant", round(float(np.mean([l[4] for l in t])), 3), "après", round(float(np.mean([l[5] for l in t])), 3),
          "décalage médian px", round(float(np.median([np.hypot(l[2], l[3]) for l in t])), 1),
          "gardées", sum(l[6] for l in lignes), "/", len(lignes), flush=True)


if __name__ == "__main__":
    main()
