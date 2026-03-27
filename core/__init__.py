"""
Core Module
===========
Owns: VLM model wrapper, image patch handling, cross-attention layer access.
Public interface consumed by the Extraction Layer:
  - VLMAuditModel   : model wrapper with hooks registered
  - AuditConfig     : dataclass holding all run-time hyperparameters
"""

from core.model import VLMAuditModel
from core.config import AuditConfig

__all__ = ["VLMAuditModel", "AuditConfig"]
