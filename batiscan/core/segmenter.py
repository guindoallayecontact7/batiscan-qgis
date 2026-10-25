"""Modèles de segmentation ONNX au format Deepness.

Les métadonnées lues dans le fichier ONNX (clés Deepness) : model_type,
class_names, resolution (cm/px), tiles_size, tiles_overlap, seg_thresh,
standardization_mean / standardization_std. Tout modèle exporté pour le
plugin Deepness (Model ZOO) fonctionne donc ici, et inversement.
"""

import json
import re

import numpy as np

from .runtime import session


def _meta(sess):
    out = {}
    for k, v in sess.get_modelmeta().custom_metadata_map.items():
        try:
            out[k] = json.loads(v)
        except (ValueError, TypeError):
            out[k] = v
    return out


class OnnxSegmenter:
    def __init__(self, path, use_gpu=True, classe=None):
        self.sess = session(path, use_gpu)
        self.meta = _meta(self.sess)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        tile = self.meta.get("tiles_size")
        if not isinstance(tile, int):
            tile = shape[-1] if isinstance(shape[-1], int) else 512
        self.tile = int(tile)
        self.resolution_m = float(self.meta.get("resolution", 50)) / 100.0
        self.overlap = float(self.meta.get("tiles_overlap", 15)) / 100.0
        self.threshold = float(self.meta.get("seg_thresh", 0.5))
        self.mean = np.array(self.meta.get("standardization_mean", [0.0, 0.0, 0.0]), np.float32)
        self.std = np.array(self.meta.get("standardization_std", [1.0, 1.0, 1.0]), np.float32)
        noms = self.meta.get("class_names", {})
        if isinstance(noms, list):
            noms = noms[0] if noms else {}
        self.class_names = {int(k): str(v) for k, v in noms.items()}
        self.classe = classe if classe is not None else self._classe_batiment()
        # Modèles BatiScan à 3 classes : la classe « contour » sépare les
        # bâtiments qui se touchent (voir TacheAuto).
        c = self.meta.get("batiscan_contour")
        self.contour = int(c) if c is not None else None

    def _classe_batiment(self):
        for k, v in self.class_names.items():
            if re.search(r"build|b[aâ]ti|house|maison|roof|toit", v, re.I):
                return k
        return 1 if len(self.class_names) != 1 else 0

    def probabilites(self, rgb):
        """Probabilités (K, H, W) pour une tuile de taille self.tile.

        K = 1 : bâtiment. K = 2 pour un modèle à contour : bâtiment (intérieur
        + contour) puis intérieur seul.
        """
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        y = self.sess.run(None, {self.input_name: x})[0][0]
        if y.shape[0] == 1:
            p = y[0]
            if p.min() < 0 or p.max() > 1:
                p = 1.0 / (1.0 + np.exp(-p))
            return p[None]
        s = y.sum(axis=0)
        if y.min() < 0 or not np.allclose(s, 1.0, atol=1e-2):
            e = np.exp(y - y.max(axis=0, keepdims=True))
            y = e / e.sum(axis=0, keepdims=True)
        if self.contour is not None:
            return np.stack([y[self.classe] + y[self.contour], y[self.classe]])
        return y[min(self.classe, y.shape[0] - 1)][None]

    def probabilites_image(self, rgb, progress=None, annule=None):
        """Cartes de probabilité pour une image de taille quelconque.

        Renvoie (bâtiment, intérieur) ; intérieur vaut None sans classe contour.
        Tuilage avec recouvrement ; seule la partie centrale de chaque tuile
        est conservée pour éviter les effets de bord.
        """
        h, w = rgb.shape[:2]
        t = self.tile
        marge = int(t * self.overlap / 2)
        pas = max(t - 2 * marge, 1)
        pad = np.zeros((h + 2 * t, w + 2 * t, 3), np.uint8)
        pad[t:t + h, t:t + w] = rgb
        out = np.zeros((2 if self.contour is not None else 1, h, w), np.float32)
        lignes = range(0, h, pas)
        colonnes = range(0, w, pas)
        total, n = len(lignes) * len(colonnes), 0
        for r in lignes:
            for c in colonnes:
                if annule and annule():
                    break
                r0, c0 = r + t - marge, c + t - marge
                p = self.probabilites(pad[r0:r0 + t, c0:c0 + t])
                hh, ww = min(pas, h - r), min(pas, w - c)
                out[:, r:r + hh, c:c + ww] = p[:, marge:marge + hh, marge:marge + ww]
                n += 1
                if progress:
                    progress(100.0 * n / total)
        return out[0], (out[1] if out.shape[0] > 1 else None)
