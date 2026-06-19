"""
neurophile.stimulus.engine
==========================
Low-latency audio playback engine (Phase 2).

Status: SCAFFOLD — Phase 2

Contributors: see CONTRIBUTING.md

Implementation notes
--------------------
- Use `sounddevice` with WASAPI Exclusive backend (Windows) or JACK (Linux)
  to achieve < 5 ms audio latency.
- Pre-load the entire audio buffer before playback starts (no streaming).
- Playback starts on a hardware interrupt, not a Python sleep timer.
- At the sample where audio begins, emit a sync event (TTL or LSL marker).

Required interface
------------------

    class StimulusEngine:
        def __init__(self, fs: int = 44100, backend: str = "wasapi"): ...
        def load(self, left: np.ndarray, right: np.ndarray) -> None:
            \"\"\"Pre-load stereo audio buffers.\"\"\"
        def play(self, sync_callback: Callable | None = None) -> float:
            \"\"\"Start playback. Returns actual start timestamp (POSIX).\"\"\"
        def stop(self) -> None: ...

Dependencies
------------
    pip install "neurophile[realtime]"   # installs sounddevice

References
----------
- sounddevice docs: https://python-sounddevice.readthedocs.io/
- WASAPI exclusive mode: https://docs.microsoft.com/en-us/windows/win32/coreaudio/wasapi
"""

raise NotImplementedError(
    "neurophile.stimulus.engine is not yet implemented. "
    "See CONTRIBUTING.md for how to contribute this module (Phase 2)."
)
