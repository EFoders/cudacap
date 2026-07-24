"""GPU rewrite of VOACAP HF propagation prediction.

Validated module-by-module against the gfortran-built voacapl oracle
(see PLAN.md and tools/oracle/). Two implementation layers:

- ``voacap_gpu.scalar``: scalar NumPy transliterations of the Fortran,
  preserving control flow and arithmetic order. Permanent cross-check layer.
- ``voacap_gpu`` proper (Stage B): vectorized PyTorch implementation.
"""

__version__ = "0.0.1"
