"""Read-only adapter for the competition photovoltaic checkpoint.

This module intentionally has no optimizer, ``fit`` or ``backward`` path.
The caller must provide the exact checkpoint produced by the PV team.  A
missing path is reported as unavailable rather than silently retraining a
replacement model on GEFCom-S.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class FrozenPVArtifactError(RuntimeError):
    pass


def load_frozen_pv_artifact(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FrozenPVArtifactError(f"frozen PV checkpoint is unavailable: {path}")
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected = metadata.get("sha256")
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise FrozenPVArtifactError("frozen PV checkpoint SHA-256 mismatch")
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=True), metadata
    except Exception as exc:
        raise FrozenPVArtifactError(f"cannot load frozen PV checkpoint: {exc}") from exc


def infer_frozen_pv(model, batch):
    """Inference-only call; gradients are disabled and no weights are changed."""
    import torch
    model.eval()
    with torch.no_grad():
        output = model(batch)
    return output
