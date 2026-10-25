"""Sessions onnxruntime partagées (un modèle n'est chargé qu'une fois)."""

import threading

_cache = {}
_lock = threading.Lock()

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")


def available_providers():
    import onnxruntime as ort
    return ort.get_available_providers()


def providers(use_gpu=True):
    dispo = available_providers()
    choix = [p for p in GPU_PROVIDERS if use_gpu and p in dispo]
    return choix + ["CPUExecutionProvider"]


def session(path, use_gpu=True):
    import onnxruntime as ort
    key = (str(path), use_gpu)
    with _lock:
        if key not in _cache:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            _cache[key] = ort.InferenceSession(str(path), sess_options=opts, providers=providers(use_gpu))
        return _cache[key]


def clear():
    with _lock:
        _cache.clear()
