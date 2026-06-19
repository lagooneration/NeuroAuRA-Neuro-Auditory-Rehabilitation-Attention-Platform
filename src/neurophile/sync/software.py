"""
neurophile.sync.software
========================
Software-only synchronization with calibration chirp correction (Tier 3).
Jitter 5–15 ms after correction. Acceptable for longitudinal tracking only.

Status: SCAFFOLD — Phase 2

Contributors: see CONTRIBUTING.md

Implementation notes
--------------------
Calibration procedure at session start:
    1. Play a 10-click chirp train (clicks at 100 ms intervals).
    2. Record EEG during the chirp.
    3. Detect the N1 peak (~100 ms post-click) in the cortical response.
    4. offset_ms = N1_latency_ms - expected_N1_latency_ms (use 95 ms as prior)
    5. Repeat chirp every 5 minutes; fit linear drift model:
         t_corrected = t_raw - (offset + drift_rate * t_elapsed_s)

Drift model
-----------
    from scipy.stats import linregress
    # offsets measured at t=[0, 300, 600, 900] seconds
    slope, intercept, *_ = linregress(times_s, measured_offsets_ms)
    # t_corrected = t_raw - (intercept + slope * t_elapsed_s)

References
----------
- Cheveigné & Simon (2008) doi:10.1016/j.jneumeth.2007.09.030
"""

raise NotImplementedError(
    "neurophile.sync.software is not yet implemented. "
    "See CONTRIBUTING.md for how to contribute this module (Phase 2)."
)
