"""
neurophile.devices.brainproducts
================================
BrainProducts LiveAmp driver (research-grade reference device).

Status: SCAFFOLD — contribute!
Sync tier: 1 (TTL) or 2 (LSL)
Channels: 32 or 64

Requirements
------------
    pip install brainvision-rda   # or use the RDA server built into BrainVision Recorder

Implementation notes
--------------------
- BrainVision Recorder must be running and configured to send data via RDA.
- The RDA (Remote Data Access) protocol streams raw EEG over TCP to localhost.
- Alternatively, the LiveAmp supports direct LSL streaming via the
  "BrainVision LSL Viewer" plugin (no BV Recorder required).
- Trigger channel is CH33 (32-ch) or CH65 (64-ch).

References
----------
- BrainProducts RDA documentation: https://www.brainproducts.com/support-resources/rda/
- LSL connector: https://github.com/brain-products/LSL-BrainVisionRDA
"""

from neurophile.devices.base import BaseDevice, DeviceInfo


class BrainProductsDevice(BaseDevice):
    """BrainProducts LiveAmp driver. SCAFFOLD — not yet implemented."""

    info = DeviceInfo(
        name="BrainProducts LiveAmp 32",
        manufacturer="Brain Products GmbH",
        n_channels=32,
        sampling_rate=1000.0,
        sync_tier=1,
        driver_package="brainvision-rda",
        notes="Research-grade reference device. Supports TTL sync. "
              "Requires BrainVision Recorder or LSL Viewer plugin.",
    )

    def connect(self) -> None:
        raise NotImplementedError(
            "BrainProductsDevice is not yet implemented. "
            "See CONTRIBUTING.md §Adding a New EEG Device Driver."
        )

    def stream(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError
