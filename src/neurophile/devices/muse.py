"""
neurophile.devices.muse
=======================
Muse 2 / Muse S consumer EEG device driver.

Status: SCAFFOLD — contribute!
Sync tier: 3 (software only)
Channels: 4 (TP9, AF7, AF8, TP10) + optional auxiliary

Requirements
------------
    pip install muselsl   # community driver, no official SDK needed

Implementation notes
--------------------
- muselsl.stream() opens a Bluetooth LE connection and starts an LSL outlet.
- Channel order: [TP9, AF7, AF8, TP10] at 256 Hz.
- Muse has no trigger input — use Tier 3 (software) sync only.
- Battery level available via auxiliary stream "Muse/elements/batt".

Pseudocode
----------
    import muselsl

    # Start streaming (this blocks; run in a subprocess or thread)
    muselsl.stream(address="00:55:DA:B9:8E:F9")

    # In parallel, record with muselsl.record() or use LabRecorder

References
----------
- muselsl: https://github.com/alexandrebarachant/muse-lsl
- Alexandre Barachant's blog: https://hackaday.io/project/162169
"""

from neurophile.devices.base import BaseDevice, DeviceInfo


class MuseDevice(BaseDevice):
    """Muse 2 / Muse S driver. SCAFFOLD — not yet implemented."""

    info = DeviceInfo(
        name="Muse 2 / Muse S",
        manufacturer="InteraXon",
        n_channels=4,
        sampling_rate=256.0,
        sync_tier=3,
        driver_package="muselsl",
        notes="Bluetooth LE. Sync Tier 3 only. Good for screening and take-home.",
    )

    def connect(self) -> None:
        raise NotImplementedError(
            "MuseDevice is not yet implemented. "
            "See CONTRIBUTING.md §Adding a New EEG Device Driver."
        )

    def stream(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError
