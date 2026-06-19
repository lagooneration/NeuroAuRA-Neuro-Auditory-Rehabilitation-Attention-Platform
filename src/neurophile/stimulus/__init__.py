"""
neurophile.stimulus
==================
Stimulus delivery, envelope extraction, CI vocoding, and paradigm scripting.

Implemented
-----------
EnvelopeExtractor   : Gammatone filterbank → Hilbert envelope
CIVocoderSimulator  : Shannon (1995) N-channel acoustic vocoder for CI simulation

Scaffold (see CONTRIBUTING.md)
------------------------------
StimulusManifest    : CC-licensed audio registry — manifest.py (contribute!)
StimulusEngine      : Low-latency audio playback — engine.py (contribute!)
ParadigmRunner      : YAML paradigm file parser — paradigm.py (contribute!)
"""

from neurophile.stimulus.envelope import EnvelopeExtractor
from neurophile.stimulus.ci_vocoder import CIVocoderSimulator

# StimulusManifest is scaffolded but not yet implemented — guard import
try:
    from neurophile.stimulus.manifest import StimulusManifest
    __all__ = ["StimulusManifest", "EnvelopeExtractor", "CIVocoderSimulator"]
except ImportError:
    __all__ = ["EnvelopeExtractor", "CIVocoderSimulator"]
