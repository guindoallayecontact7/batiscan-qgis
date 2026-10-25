"""Étape 4 : export du modèle entraîné en ONNX, avec les métadonnées Deepness.

python 04_exporter_onnx.py <meilleur.pt> <modele.onnx> [--tuile 512] [--resolution-cm 30]

Le fichier produit se charge dans BatiScan (onglet Modèles > Ajouter un
modèle .onnx) comme dans le plugin Deepness.
"""

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poids")
    ap.add_argument("sortie")
    ap.add_argument("--tuile", type=int, default=512)
    ap.add_argument("--resolution-cm", type=float, default=30)
    a = ap.parse_args()

    ck = torch.load(a.poids, map_location="cpu", weights_only=False)
    modele = smp.Unet(ck["encodeur"], encoder_weights=None, classes=3)
    modele.load_state_dict(ck["etat"])
    modele.eval()

    class AvecSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).softmax(1)

    net = AvecSoftmax(modele).eval()
    x = torch.randn(1, 3, a.tuile, a.tuile)
    torch.onnx.export(net, x, a.sortie, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamo=False)

    m = onnx.load(a.sortie)
    meta = {
        "model_type": "Segmentor",
        "class_names": {"0": "Fond", "1": "Bâtiment", "2": "Contour"},
        "resolution": a.resolution_cm,
        "tiles_size": a.tuile,
        "tiles_overlap": 25,
        "seg_thresh": 0.5,
        "seg_small_segment": 9,
        "standardization_mean": [0.485, 0.456, 0.406],
        "standardization_std": [0.229, 0.224, 0.225],
        "batiscan_contour": 2,
        "batiscan_metriques_validation": ck.get("metriques", {}),
    }
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key = k
        p.value = json.dumps(v, ensure_ascii=False)
    onnx.save(m, a.sortie)

    net.eval()  # l'export remet le module dans son mode d'origine
    s = ort.InferenceSession(a.sortie, providers=["CPUExecutionProvider"])
    y = s.run(None, {"input": x.numpy()})[0]
    with torch.no_grad():
        ref = net(x).numpy()
    print("écart PyTorch/ONNX :", float(np.abs(y - ref).max()), "sortie", y.shape)


if __name__ == "__main__":
    main()
