"""Mesure par village : F1 pixel et rappel/précision par bâtiment, sur les vignettes d'un jeu.

python 07_mesurer.py <jeu> <village,village> <modele1.pt> [<modele2.pt> ...]

Un bâtiment de référence est retrouvé si au moins 50 % de sa surface est prédite.
Une détection est juste si au moins 30 % de sa surface tombe sur la référence.
Une référence incomplète fait baisser la précision sans que le modèle soit en cause.
"""
import glob, os, sys
import cv2, numpy as np, segmentation_models_pytorch as smp, torch
MOY = np.array([0.485, 0.456, 0.406], np.float32); ECT = np.array([0.229, 0.224, 0.225], np.float32)


def objets(b):
    n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=4)
    return n, lab


def main():
    jeu, vs, mods = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
    print("modele;village;tuiles;F1_pixel;P_pixel;R_pixel;ref;rappel_bat;detections;precision_bat")
    for mp in mods:
        ck = torch.load(mp, map_location="cuda", weights_only=False)
        net = smp.Unet(ck["encodeur"], encoder_weights=None, classes=3).cuda().eval(); net.load_state_dict(ck["etat"])
        for v in vs:
            tp = fp = fn = ref = trouve = det = juste = n = 0
            for d in glob.glob(os.path.join(jeu, v + "_*")):
                for f in glob.glob(os.path.join(d, "img_*.png")):
                    m = cv2.imread(f.replace("img_", "msk_"), cv2.IMREAD_GRAYSCALE)
                    x = (cv2.imread(f)[..., ::-1].astype(np.float32) / 255 - MOY) / ECT
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        c = net(torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda())[0].argmax(0).cpu().numpy()
                    val = m != 255; g = (m > 0) & val; p = (c > 0) & val; n += 1
                    tp += (p & g).sum(); fp += (p & ~g).sum(); fn += (~p & g).sum()
                    ng, lg = objets(m == 1); np_, lp = objets(c == 1)
                    gd = cv2.dilate((lg > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    for i in range(1, ng):
                        o = lg == i
                        if o.sum() < 20: continue
                        ref += 1; trouve += p[o].mean() >= 0.5
                    for i in range(1, np_):
                        o = (lp == i) & val
                        if o.sum() < 20: continue
                        det += 1; juste += (g | gd)[o].mean() >= 0.3
            P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
            print(f"{os.path.dirname(mp)};{v};{n};{2*P*R/max(P+R,1e-9):.3f};{P:.3f};{R:.3f};{ref};{trouve/max(ref,1):.3f};{det};{juste/max(det,1):.3f}", flush=True)


if __name__ == "__main__":
    main()
