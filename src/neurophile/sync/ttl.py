"""
neurophile.sync.ttl
==================
Hardware TTL synchronization (Tier 1). Jitter < 0.5 ms.

Status: SCAFFOLD — Phase 2

Contributors: see CONTRIBUTING.md §Real-time Pipeline

Implementation notes
--------------------
- Send a 5 V, 1 ms pulse on parallel port (LPT) or BNC breakout at the exact
  audio sample where stimulus playback begins.
- Record the pulse as a trigger channel in the EEG amplifier.
- Post-hoc: detect TTL rising edge in the EEG trigger channel; use the
  corresponding sample index as the ground-truth stimulus onset.

Pseudocode
----------
    port = parallel.ParallelPort(address=0x0378)   # Windows LPT1
    port.setData(1)       # pulse high
    time.sleep(0.001)
    port.setData(0)       # pulse low

Required hardware
-----------------
- Parallel port adapter or USB-to-parallel (e.g. StarTech ICUSB1284)
- BNC splitter if sharing with EEG amplifier
- Amplifier with trigger / DIO input (all research-grade amps support this)

References
----------
- Picton et al. (2000) Guidelines for using EEG to study cognition.
- PsychoPy parallel port documentation: https://psychopy.org/api/parallel.html
"""

raise NotImplementedError(
    "neurophile.sync.ttl is not yet implemented. "
    "See CONTRIBUTING.md for how to contribute this module (Phase 2)."
)
