"""Portable source, weight, precision and JSON utilities; no experiment imports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import sys
import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"
WHEEL_SHA = "43986f644c383f255e73dcbd2305e311f15fd70b4aeec54deb346dab5d5cbfd7"
METRIX_SHA = "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)

def environment():
    names = ("torch", "torchvision", "numpy", "scipy", "quantus", "pandas", "captum", "kaggle", "Pillow")
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"python": sys.version, "interpreter": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions,
            "torch_cuda_version": torch.version.cuda,
            "inside_container": Path("/.dockerenv").exists()}

def model_state_digest(model):
    """Current-run identity of every parameter and buffer, including .data edits.

    Tensor values are copied to CPU only for hashing, without model inference or
    files. Version counters alone cannot detect edits through parameter.data.
    """
    digest = hashlib.sha256()
    for kind, tensors in (("parameter", model.named_parameters()),
                          ("buffer", model.named_buffers())):
        for name, tensor in tensors:
            try: version = tensor._version
            except RuntimeError: version = None
            metadata = (kind, name, tuple(tensor.shape), str(tensor.dtype),
                        str(tensor.device), tensor.data_ptr(), version)
            digest.update(json.dumps(metadata, separators=(",", ":")).encode())
            value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()

def official_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def selected_device(name):
    if name == "auto": name = official_device()
    device = torch.device(name)
    if device.type == "mps":
        if not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "", "0"):
            raise RuntimeError("Disable MPS CPU fallback before starting Python")
    elif device.type == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        # New PyTorch precision controls; do not mix them with allow_tf32 setters.
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    elif device.type != "cpu": raise ValueError("Expected cpu, mps, cuda or auto")
    return device

def verify_sources(weights):
    weights = Path(weights)
    if weights.stat().st_size != 282678575 or sha(weights) != WEIGHT_SHA:
        raise RuntimeError("Official weight size or SHA-256 differs")
    wheel = MODEL_ROOT / "dist/kaggle-0.1.0-py3-none-any.whl"
    if sha(wheel) != WHEEL_SHA: raise RuntimeError("Local official wheel identity differs")
    scorer = MODEL_ROOT / "src/official_metrix.py"
    if sha(scorer) != METRIX_SHA: raise RuntimeError("Official scorer identity differs")
    from kaggle.submmit import submmit
    package = Path(submmit.__file__).resolve().parents[1]
    expected = json.loads((MODEL_ROOT / "official_sources.json").read_text())
    for relative, digest in expected["package_file_sha256"].items():
        if sha(package / relative) != digest:
            raise RuntimeError(f"Installed official wheel Python source differs: {relative}")
    if importlib.metadata.version("kaggle") != "0.1.0":
        raise RuntimeError("Install the supplied wheel, not the PyPI Kaggle API package")
    return {"weights_sha256": WEIGHT_SHA, "wheel_sha256": WHEEL_SHA,
            "metrix_sha256": METRIX_SHA,
            "installed_package_root": str(package), "official_sources": "PASS"}
