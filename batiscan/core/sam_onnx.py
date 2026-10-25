"""Segment Anything (SAM, MobileSAM) exécuté avec onnxruntime, sans PyTorch.

Deux formats d'encodeur sont pris en charge :
- entrée (H, W, 3) en float RVB brut, prétraitement inclus dans le graphe
  (exports « AnyLabeling » de vietanhdev/segment-anything-onnx-models). Ce
  graphe travaille sur un canevas fixe de 1024 x 682 px : l'image doit y être
  posée en haut à gauche, sinon les masques sortent étirés ;
- entrée (1, 3, 1024, 1024) normalisée (export SAM classique).
Le décodeur est l'export ONNX officiel de SAM (point_coords, point_labels...).

Les bâtiments ruraux font 10 à 20 px à 0,3 m. SAM les ignore dans une grande
fenêtre et renvoie alors toute l'image : on lui passe donc de petites vignettes
(128 à 256 px) agrandies sur le canevas.
"""

import cv2
import numpy as np

from .runtime import session

_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)


class SamOnnx:
    def __init__(self, encoder_path, decoder_path, canvas=(682, 1024), use_gpu=True):
        self.enc = session(encoder_path, use_gpu)
        self.dec = session(decoder_path, use_gpu)
        inp = self.enc.get_inputs()[0]
        self.enc_input = inp.name
        self.raw_hwc = len(inp.shape) == 3
        self.canvas = tuple(canvas) if self.raw_hwc else (1024, 1024)
        self.embedding = None

    # -- encodage ---------------------------------------------------------
    def set_image(self, rgb):
        """Calcule l'empreinte de l'image (étape lente, une fois par vignette)."""
        h, w = rgb.shape[:2]
        ch, cw = self.canvas
        s = min(ch / h, cw / w)
        fh, fw = int(round(h * s)), int(round(w * s))
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        img = cv2.resize(rgb, (fw, fh), interpolation=interp).astype(np.float32)
        if not self.raw_hwc:
            img = (img - _MEAN) / _STD
        pad = np.zeros((ch, cw, 3), np.float32)
        pad[:fh, :fw] = img
        x = pad if self.raw_hwc else pad.transpose(2, 0, 1)[None]
        self.orig_size, self.fed_size, self.scale = (h, w), (fh, fw), s
        self.embedding = self.enc.run(None, {self.enc_input: x})[0]

    # -- décodage ---------------------------------------------------------
    def predict(self, points, labels):
        """Masques candidats pour des points en pixels de l'image d'origine.

        points : liste de (col, ligne) ; labels : 1 = bâtiment, 0 = à exclure.
        Renvoie une liste de (masque booléen (H, W), score) triée par score.
        """
        coords = np.asarray(points, np.float32).reshape(-1, 2) * self.scale
        labs = np.asarray(labels, np.float32).reshape(-1)
        # point de remplissage (-1) attendu par l'export SAM sans boîte
        coords = np.concatenate([coords, np.zeros((1, 2), np.float32)])[None]
        labs = np.concatenate([labs, np.array([-1], np.float32)])[None]
        fh, fw = self.fed_size
        out_size = self.canvas if self.raw_hwc else self.fed_size
        masks, scores, _ = self.dec.run(None, {
            "image_embeddings": self.embedding,
            "point_coords": coords,
            "point_labels": labs,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array(out_size, np.float32),
        })
        h, w = self.orig_size
        out = []
        for k in range(masks.shape[1]):
            logit = masks[0, k, :fh, :fw]
            if logit.shape != (h, w):
                logit = cv2.resize(logit, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append((logit > 0.0, float(scores[0, k])))
        out.sort(key=lambda m: -m[1])
        return out
