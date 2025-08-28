"""
Utility helpers used by scripts (logging, hashes, etc.).

Exports:
- log_breakdown(): append a breakdown row to results CSV
- file_sha256():   short (8-char) SHA-256 of a file for reproducibility
"""

from .logger import log_breakdown, file_sha256

__all__ = ["log_breakdown", "file_sha256"]


