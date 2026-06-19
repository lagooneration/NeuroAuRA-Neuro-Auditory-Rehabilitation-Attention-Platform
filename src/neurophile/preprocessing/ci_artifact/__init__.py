"""
neurophile.preprocessing.ci_artifact
=====================================
Three-stage cochlear implant artifact rejection pipeline.

Stage   Module                  Status
------  ----------------------  ----------------------------
1       template_subtraction    ✅ Implemented
2       spatial_filter          🔧 Scaffold — contribute!
3       ica_cancellation        ✅ Implemented (ICA-based)

The pipeline orchestrator (pipeline.py) runs whichever stages are enabled
in the YAML config. Disabled stages pass data through unchanged.
"""

from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline
from neurophile.preprocessing.ci_artifact.quality import CIArtifactQuality
from neurophile.preprocessing.ci_artifact.ica_cancellation import ICACancellation

__all__ = ["CIArtifactPipeline", "CIArtifactQuality", "ICACancellation"]
