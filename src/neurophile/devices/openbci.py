"""
neurophile.devices.openbci
==========================
OpenBCI Cyton (8-ch) and Ganglion (4-ch) device driver via BrainFlow.

Sync tier: 2 (LSL) — the driver streams to an LSL outlet which LabRecorder
or the neurophile real-time pipeline can subscribe to.

Requirements
------------
    pip install "neurophile[openbci]"   # installs brainflow

Usage
-----
    from neurophile.devices.openbci import OpenBCIDevice

    device = OpenBCIDevice(board="cyton", port="COM3")
    device.connect()
    device.stream()       # blocks; Ctrl-C to stop
    device.disconnect()

    # Or as a context manager:
    with OpenBCIDevice(board="cyton", port="COM3") as device:
        device.stream()
"""

from __future__ import annotations

import logging
import threading
import time

from neurophile.devices.base import BaseDevice, DeviceInfo

logger = logging.getLogger(__name__)

# ── BrainFlow board IDs ────────────────────────────────────────────────────────
_BOARD_IDS: dict[str, int] = {
    "cyton": 0,       # OpenBCI Cyton (8 ch, 250 Hz)
    "ganglion": 1,    # OpenBCI Ganglion (4 ch, 200 Hz)
    "cyton_daisy": 2, # Cyton + Daisy (16 ch, 125 Hz)
}

_BOARD_INFO: dict[str, DeviceInfo] = {
    "cyton": DeviceInfo(
        name="OpenBCI Cyton",
        manufacturer="OpenBCI",
        n_channels=8,
        sampling_rate=250.0,
        sync_tier=2,
        driver_package="brainflow",
        notes="Connect via USB dongle. Serial port typically COM3 (Win) or /dev/ttyUSB0 (Linux).",
    ),
    "ganglion": DeviceInfo(
        name="OpenBCI Ganglion",
        manufacturer="OpenBCI",
        n_channels=4,
        sampling_rate=200.0,
        sync_tier=2,
        driver_package="brainflow",
        notes="Bluetooth LE. Use for screening; 4 ch is insufficient for full AAD decoding.",
    ),
    "cyton_daisy": DeviceInfo(
        name="OpenBCI Cyton + Daisy",
        manufacturer="OpenBCI",
        n_channels=16,
        sampling_rate=125.0,
        sync_tier=2,
        driver_package="brainflow",
        notes="16-ch configuration. Sampling rate halved to 125 Hz; resample to EEG pipeline rate.",
    ),
}


class OpenBCIDevice(BaseDevice):
    """OpenBCI Cyton / Ganglion driver via BrainFlow + LSL outlet.

    Parameters
    ----------
    board : {"cyton", "ganglion", "cyton_daisy"}
        Board type.
    port : str
        Serial port (Cyton) or MAC address (Ganglion via BLE).
        Example: ``"COM3"`` on Windows, ``"/dev/ttyUSB0"`` on Linux.
    lsl_stream_name : str
        Name of the LSL outlet. Downstream tools subscribe to this name.
    """

    def __init__(
        self,
        board: str = "cyton",
        port: str = "COM3",
        lsl_stream_name: str = "OpenBCI_EEG",
    ) -> None:
        if board not in _BOARD_IDS:
            raise ValueError(
                f"Unknown board {board!r}. Choose from {list(_BOARD_IDS)}"
            )
        self._board_name = board
        self._port = port
        self._lsl_stream_name = lsl_stream_name
        self._board = None
        self._outlet = None
        self._streaming = False
        self._thread: threading.Thread | None = None

    @property
    def info(self) -> DeviceInfo:  # type: ignore[override]
        return _BOARD_INFO[self._board_name]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open BrainFlow session and initialise the LSL outlet."""
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
            from pylsl import StreamInfo, StreamOutlet
        except ImportError as exc:
            raise ImportError(
                "OpenBCI support requires brainflow and pylsl. "
                "Install with: pip install 'neurophile[openbci,realtime]'"
            ) from exc

        params = BrainFlowInputParams()
        params.serial_port = self._port

        board_id = _BOARD_IDS[self._board_name]
        self._board = BoardShim(board_id, params)
        self._board.prepare_session()

        # Create LSL outlet
        lsl_info = StreamInfo(
            name=self._lsl_stream_name,
            type="EEG",
            channel_count=self.info.n_channels,
            nominal_srate=self.info.sampling_rate,
            channel_format="float32",
            source_id=f"openbci_{self._board_name}_{self._port}",
        )
        # Attach channel labels to LSL metadata
        chans = lsl_info.desc().append_child("channels")
        for i in range(self.info.n_channels):
            ch = chans.append_child("channel")
            ch.append_child_value("label", f"CH{i+1}")
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

        self._outlet = StreamOutlet(lsl_info)
        logger.info(
            "Connected to %s on %s. LSL outlet: %s",
            self.info.name, self._port, self._lsl_stream_name,
        )

    def stream(self) -> None:
        """Start BrainFlow acquisition and push samples to the LSL outlet.

        Blocks until :meth:`disconnect` is called or KeyboardInterrupt.
        """
        if self._board is None or self._outlet is None:
            raise RuntimeError("Call connect() before stream().")

        from brainflow.board_shim import BoardShim

        self._board.start_stream()
        self._streaming = True
        logger.info("Streaming started. Press Ctrl-C to stop.")

        board_id = _BOARD_IDS[self._board_name]
        eeg_channels = BoardShim.get_eeg_channels(board_id)

        try:
            while self._streaming:
                data = self._board.get_board_data()  # (n_ch, n_samples)
                if data.shape[1] == 0:
                    time.sleep(0.004)   # ~4 ms poll; BrainFlow buffers 4x
                    continue
                # Push each sample to LSL
                eeg = data[eeg_channels, :].T.tolist()  # list of lists
                for sample in eeg:
                    self._outlet.push_sample(sample)
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user.")
        finally:
            self._board.stop_stream()

    def disconnect(self) -> None:
        """Stop streaming and release the BrainFlow session."""
        self._streaming = False
        if self._board is not None:
            try:
                self._board.release_session()
            except Exception:
                pass
        self._board = None
        self._outlet = None
        logger.info("Disconnected from %s.", self.info.name)

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "OpenBCIDevice":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()


# ── Scaffold stubs for other OpenBCI boards ───────────────────────────────────

class _ScaffoldDevice(BaseDevice):
    """Internal: base for unimplemented device scaffolds."""

    def connect(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} is not yet implemented. "
            "See CONTRIBUTING.md §Adding a New EEG Device Driver."
        )

    def stream(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError
