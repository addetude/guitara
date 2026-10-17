"""Live audio from your amp / audio interface, kept in a ring buffer for the transcriber."""

import threading
import time

import numpy as np


class AudioInput:
    def __init__(self, device: int | str | None = None, sample_rate: int = 22050, seconds: float = 4.0, channel: int = 0):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as err:  # pragma: no cover - needs PortAudio
            raise RuntimeError("Live audio needs sounddevice + PortAudio: pip install 'guitra[audio]'") from err
        self.sr = sample_rate
        self._buf = np.zeros(int(seconds * sample_rate), dtype=np.float32)
        self._lock = threading.Lock()
        self._channel = channel
        self._written_until = 0.0  # monotonic time of the newest sample
        info = sd.query_devices(device, "input")
        self._stream = sd.InputStream(
            device=device,
            channels=min(int(info["max_input_channels"]), channel + 1),
            samplerate=sample_rate,
            blocksize=256,
            latency="low",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # runs on the audio thread
        mono = indata[:, self._channel].astype(np.float32)
        with self._lock:
            self._buf = np.roll(self._buf, -frames)
            self._buf[-frames:] = mono
            self._written_until = time.monotonic()

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def latest(self, seconds: float) -> tuple[np.ndarray, float]:
        """The last `seconds` of audio and the monotonic time its first sample was captured."""
        n = int(seconds * self.sr)
        with self._lock:
            chunk = self._buf[-n:].copy()
            end = self._written_until
        return chunk, end - n / self.sr

    def level(self, seconds: float = 0.1) -> float:
        chunk, _ = self.latest(seconds)
        return float(np.sqrt(np.mean(chunk**2)))


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())
