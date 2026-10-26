"""Trimodal (sRGB + thermal + methane) produce-spoilage prediction on the TR-6 dataset.

Reusable code behind the scripts in ``scripts/``; the notebooks in ``Code/`` are the original
research record and are kept unchanged.
"""
from .config import VARIANTS, Paths, TrainConfig  # noqa: F401

__version__ = "1.0.0"
