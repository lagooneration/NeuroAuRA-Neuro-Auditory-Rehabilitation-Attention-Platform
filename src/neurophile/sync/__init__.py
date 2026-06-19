"""
neurophile.sync
==============
Temporal synchronization between audio stimulus delivery and EEG recording.

Tiers
-----
Tier 1  ttl.py          Hardware TTL trigger via parallel port / BNC  🔧 Phase 2
Tier 2  lsl.py          Lab Streaming Layer network sync              🔧 Phase 2
Tier 3  software.py     Software timestamp + calibration chirps       🔧 Phase 2
        drift.py        Clock drift estimation and correction          🔧 Phase 2
        quality.py      Alignment quality grading (A–F)               ✅ in data.validators.alignment
"""
