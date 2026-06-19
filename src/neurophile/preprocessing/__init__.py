"""
neurophile.preprocessing
========================
Signal processing pipelines.

Implemented
-----------
StandardPipeline        : MNE-based offline pipeline for normal-hearing EEG
CIArtifactPipeline      : Three-stage CI artifact rejection (Stage 1 complete)

Scaffold (see CONTRIBUTING.md)
------------------------------
ICAWrapper              : CI-aware ICA — src/neurophile/preprocessing/ica.py
"""

from neurophile.preprocessing.standard import StandardPipeline
from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline

__all__ = ["StandardPipeline", "CIArtifactPipeline"]
