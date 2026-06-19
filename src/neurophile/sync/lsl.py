"""
neurophile.sync.lsl
==================
Lab Streaming Layer (LSL) network synchronization (Tier 2). Jitter 1–3 ms.

Status: SCAFFOLD — Phase 2

Contributors: see CONTRIBUTING.md §Real-time Pipeline

Implementation notes
--------------------
- Both stimulus PC and EEG PC must run the LSL clock sync daemon.
- The stimulus engine creates two LSL outlets:
    "AudioMarkers"  — string stream, one sample per stimulus event
    "AudioEnvelope" — float32 stream @ EEG sampling rate
- LabRecorder records all streams into a single .xdf file.
- On loading, pyxdf.load_xdf() returns each stream with its clock-corrected
  timestamps. Use these to align EEG to stimulus.

Key pyxdf usage
---------------
    import pyxdf
    streams, header = pyxdf.load_xdf("recording.xdf")
    eeg_stream = next(s for s in streams if s["info"]["type"][0] == "EEG")
    marker_stream = next(s for s in streams if s["info"]["name"][0] == "AudioMarkers")

    eeg_ts = eeg_stream["time_stamps"]        # already clock-corrected
    marker_ts = marker_stream["time_stamps"]  # already clock-corrected
    # Align: find closest EEG sample for each marker timestamp

References
----------
- Kothe & Jung (2016) https://labstreaminglayer.org
- pyxdf: https://github.com/xdf-modules/pyxdf
- Cheveigné et al. (2018) doi:10.1016/j.jneumeth.2018.01.015
"""

raise NotImplementedError(
    "neurophile.sync.lsl is not yet implemented. "
    "See CONTRIBUTING.md for how to contribute this module (Phase 2)."
)
