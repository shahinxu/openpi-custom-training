"""Minimal PyTorch model stubs for OpenPI.

The current project only uses the JAX implementations of OpenPI models,
so we provide a lightweight placeholder to satisfy imports. If PyTorch
models are needed in the future, replace these stubs with the actual
implementations from the upstream OpenPI repository.
"""

from .pi0_pytorch import PI0Pytorch

__all__ = ["PI0Pytorch"]
