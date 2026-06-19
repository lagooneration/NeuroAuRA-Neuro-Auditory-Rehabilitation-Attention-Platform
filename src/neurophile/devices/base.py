"""
neurophile.devices.base
=======================
Abstract base protocol that all EEG device drivers must implement.

Adding a new device
-------------------
1. Subclass BaseDevice in a new file (e.g. devices/muse.py).
2. Populate a DeviceInfo dataclass on the class.
3. Implement connect(), stream(), and disconnect().
4. Register in devices/__init__.py.
5. See CONTRIBUTING.md §Adding a New EEG Device Driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Metadata describing a supported EEG device."""

    name: str
    manufacturer: str
    n_channels: int
    sampling_rate: float  # Hz
    sync_tier: int  # 1=TTL, 2=LSL, 3=software
    driver_package: str  # pip install name
    notes: str = ""


class BaseDevice(ABC):
    """Protocol that every EEG device driver must implement.

    The driver is responsible for:
    - Connecting to the hardware.
    - Starting an LSL outlet that streams EEG data.
    - Reporting the device info (channel names, sampling rate, etc.).

    The rest of the pipeline is device-agnostic; it reads from the LSL outlet.
    """

    #: Subclasses must define this at the class level.
    info: DeviceInfo

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the device."""

    @abstractmethod
    def stream(self) -> None:
        """Start streaming EEG data to an LSL outlet.

        This method blocks (or starts a background thread) until
        :meth:`disconnect` is called.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Stop streaming and close the device connection."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.info.name!r}, "
            f"n_ch={self.info.n_channels}, "
            f"fs={self.info.sampling_rate} Hz)"
        )
