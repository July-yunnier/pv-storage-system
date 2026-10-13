"""Frozen PV inference release. No fitting or model selection entry points."""

from .infer import FrozenPVModel, load_model

__all__ = ["FrozenPVModel", "load_model"]
