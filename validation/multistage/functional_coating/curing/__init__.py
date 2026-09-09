"""Standalone thermal-curing stage physics model."""

from .physics_model import CONTROL_BOUNDS, INCOMING_STATE_BOUNDS, evaluate_model

__all__ = ["CONTROL_BOUNDS", "INCOMING_STATE_BOUNDS", "evaluate_model"]
