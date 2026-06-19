"""
Neurophile — Neuro-Auditory Rehabilitation & Attention Platform
==============================================================

Subpackages
-----------
data          : BIDS-EEG read/write, HDF5 streaming, metadata validation
sync          : Temporal synchronization (TTL / LSL / software)
stimulus      : Stimulus delivery, envelope extraction, paradigm scripting
devices       : EEG device drivers (OpenBCI implemented; others scaffolded)
preprocessing : Signal processing pipelines (standard EEG + CI artifact)
decoding      : AAD decoders and parallel evaluation harness
federated     : Federated learning (edge + server) — Phase 4 scaffold
visualization : Dashboards and session reports — Phase 3 scaffold
cli           : `neurophile` command-line interface
"""

from neurophile._version import __version__

__all__ = ["__version__"]
