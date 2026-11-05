"""Repère les fausses détections d'un modèle sur un jeu et les marque comme négatifs difficiles.

python 12_negatifs.py <modele.pt> <jeu> <jeu_sortie> <planche.png> --exclure Dialakoro Doumanani

Le jeu d'origine n'est pas modifié. Dans <jeu_sortie>, les masques gagnent la valeur 3 :
du fond que le modèle prend pour du bâti (cour, sol nu). Un objet prédit compte si moins
de 30 % de sa surface tombe sur la référence et si moins de la moitié est ignorée.
Les villages exclus sont copiés tels quels : ils servent à la validation.
"""
import argparse, glob, os, random, shutil
import cv2, numpy as np, segmentation_models_pytorch as smp, torch
MOY = np.array([0.485, 0.456, 0.406], np.float32); ECT = np.array([0.229, 0.224, 0.225], np.float32)
AIRE_MIN = 89  # pixels à 0,3 m, soit 8 m²


def contraste(lab, z):
    """Écart de couleur Lab entre le cœur de l'objet et un anneau de 0,6 à 2,7 m."""
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    u = z.astype(np.uint8)
    coeur = cv2.erode(u, k(2)); coeur = coeur if coeur.sum() >= 20 else u
    anneau = (cv2.dilate(u, k(9)) - cv2.dilate(u, k(2))) > 0
    return float(np.linalg.norm(np.median(lab[coeur > 0], 0) - np.median(lab[anneau], 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modele"); ap.add_argument("jeu"); ap.add_argument("sortie"); ap.add_argument("planche")
    ap.add_argument("--exclure", nargs="*", default=[])
    ap.add_argument("--de-max", type=float, default=20, help="sous ce contraste : sol nu, négatif ; au-dessus : toit probable, ignoré")
    a = ap.parse_args()
    ck = torch.load(a.modele, map_location="cuda", weights_only=False)
    net = smp.Unet(ck["encodeur"], encoder_weights=None, classes=3).cuda().eval(); net.load_state_dict(ck["etat"])
    total = marques = tuiles = toits = 0
    cases = []
    for d in sorted(glob.glob(os.path.join(a.jeu, "*"))):
        nom = os.path.basename(d); village = nom.rsplit("_", 1)[0]
        o = os.path.join(a.sortie, nom); os.makedirs(o, exist_ok=True)
        n = 0
        for f in sorted(glob.glob(os.path.join(d, "img_*.png"))):
            fm = f.replace("img_", "msk_")
            shutil.copy(f, o)
            m = cv2.imread(fm, cv2.IMREAD_GRAYSCALE)
            if village not in a.exclure:
                im = cv2.imread(f)
                x = (im[..., ::-1].astype(np.float32) / 255 - MOY) / ECT
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    c = net(torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda())[0].argmax(0).cpu().numpy()
                k, lab, st, ce = cv2.connectedComponentsWithStats((c > 0).astype(np.uint8), connectivity=4)
                ref = (m == 1) | (m == 2)
                lab_im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
                sortie = m.copy()
                for i in range(1, k):
                    if st[i, cv2.CC_STAT_AREA] < AIRE_MIN:
                        continue
                    z = lab == i
                    total += 1
                    if ref[z].mean() < 0.3 and (m[z] == 255).mean() < 0.5:
                        if contraste(lab_im, z) >= a.de_max:
                            sortie[z & (m == 0)] = 255  # toit probable oublié par la référence
                            toits += 1
                            continue
                        sortie[z & (m == 0)] = 3
                        marques += 1; n += 1
                        cx, cy = ce[i].astype(int)
                        r = 50; p = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT)
                        v = p[cy:cy + 2 * r, cx:cx + 2 * r].copy()
                        q = cv2.copyMakeBorder(z.astype(np.uint8), r, r, r, r, cv2.BORDER_CONSTANT)[cy:cy + 2 * r, cx:cx + 2 * r]
                        cs, _ = cv2.findContours(q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(v, cs, -1, (0, 255, 255), 1)
                        cases.append(v)
                m = sortie
            cv2.imwrite(os.path.join(o, os.path.basename(fm)), m)
            tuiles += 1
        print(nom, n, flush=True)
    print(f"{tuiles} vignettes ; {total} objets prédits ; {marques} négatifs difficiles ({100 * marques / max(total, 1):.1f} %) ; {toits} toits probables ignorés")
    random.seed(1); random.shuffle(cases); cases = cases[:96]
    cases += [np.zeros((100, 100, 3), np.uint8)] * (-len(cases) % 12)
    if cases:
        cv2.imwrite(a.planche, np.vstack([np.hstack(cases[i:i + 12]) for i in range(0, len(cases), 12)]))


if __name__ == "__main__":
    main()
