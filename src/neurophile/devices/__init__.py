"""
neurophile.devices
=================
EEG device drivers. All drivers produce an LSL outlet so the rest of the
pipeline is device-agnostic.

Implemented
-----------
OpenBCIDevice       : Cyton (8-ch) and Ganglion (4-ch) via brainflow

Scaffold (see CONTRIBUTING.md)
------------------------------
MuseDevice          : Muse 2 / Muse S via muselsl
BrainProductsDevice : LiveAmp via brainvision-rda
GtecDevice          : g.USBamp via pygds
EmotivDevice        : EPOC X via Cortex SDK
LSLGenericDevice    : Any LSL-compatible device (passthrough)

All devices must implement BaseDevice (base.py).
"""

from neurophile.devices.base import BaseDevice, DeviceInfo
from neurophile.devices.openbci import OpenBCIDevice

__all__ = ["BaseDevice", "DeviceInfo", "OpenBCIDevice"]
