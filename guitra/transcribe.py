"""Audio → notes.

`BasicPitchTranscriber` uses Spotify's Basic Pitch (polyphonic: chords work). It's an optional
dependency (`pip install guitra[transcribe]`) because it pulls in a large ML runtime.
`YinTranscriber` is a small built-in fallback for single notes (scales, riffs, bass lines).
"""

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .notes import hz_to_midi, midi_to_hz


@dataclass(frozen=True)
class NoteEvent:
    midi: int
    start: float  # seconds (same clock as the camera loop)
    end: float
    amplitude: float  # 0..1


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray, sr: int, t0: float) -> list[NoteEvent]: ...


# --------------------------------------------------------------------------------------------
# YIN (de Cheveigné & Kawahara, 2002): monophonic pitch detection


def yin_pitch(frame: np.ndarray, sr: int, fmin: float, fmax: float, threshold: float = 0.12) -> float | None:
    """Fundamental frequency of one frame in Hz, or None if unvoiced."""
    x = frame.astype(np.float64) - np.mean(frame)
    n = len(x)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(n // 2, int(sr / fmin) + 1)
    if tau_max <= tau_min:
        return None
    w = n - tau_max
    # difference function d(tau) = sum_j (x_j - x_{j+tau})^2 over a window of length w
    energy = np.cumsum(np.concatenate([[0.0], x * x]))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, size)
    seg = np.fft.rfft(x[:w], size)
    corr = np.fft.irfft(np.conj(seg) * spec, size)[: tau_max + 1]  # sum_j x_j x_{j+tau}, j < w
    taus = np.arange(tau_max + 1)
    d = energy[w] + (energy[taus + w] - energy[taus]) - 2 * corr
    # cumulative mean normalized difference
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(running, 1e-12)
    below = np.where(cmnd[tau_min:] < threshold)[0]
    if len(below) == 0:
        return None
    tau = tau_min + below[0]
    while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:  # walk down to the local minimum
        tau += 1
    if 1 <= tau < tau_max:  # parabolic interpolation
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a - 2 * b + c
        tau = tau + 0.5 * (a - c) / denom if denom != 0 else tau
    return sr / tau


class YinTranscriber:
    def __init__(self, lowest_midi: int, highest_midi: int, frame: int = 2048, hop: int = 512, min_rms: float = 0.01):
        self.fmin = midi_to_hz(lowest_midi - 1)
        self.fmax = midi_to_hz(highest_midi + 1)
        self.frame, self.hop, self.min_rms = frame, hop, min_rms

    def transcribe(self, samples: np.ndarray, sr: int, t0: float) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        cur: tuple[int, float, float] | None = None  # (midi, start, peak)
        for i in range(0, max(0, len(samples) - self.frame + 1), self.hop):
            frame = samples[i : i + self.frame]
            rms = float(np.sqrt(np.mean(frame**2)))
            hz = yin_pitch(frame, sr, self.fmin, self.fmax) if rms >= self.min_rms else None
            t = t0 + i / sr
            midi = int(round(hz_to_midi(hz))) if hz else None
            if cur and midi != cur[0]:
                events.append(NoteEvent(cur[0], cur[1], t, min(1.0, cur[2])))
                cur = None
            if midi is not None:
                cur = (midi, t, rms * 4) if cur is None else (cur[0], cur[1], max(cur[2], rms * 4))
        if cur:
            events.append(NoteEvent(cur[0], cur[1], t0 + len(samples) / sr, min(1.0, cur[2])))
        min_len = 2.5 * self.hop / sr  # drop blips shorter than ~3 frames
        return [e for e in events if e.end - e.start >= min_len]


# --------------------------------------------------------------------------------------------
# Spotify Basic Pitch: polyphonic transcription


class BasicPitchTranscriber:
    def __init__(self, lowest_midi: int, highest_midi: int, onset_threshold: float = 0.5, frame_threshold: float = 0.3):
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict
        except ImportError as err:  # pragma: no cover - depends on optional install
            raise RuntimeError("Basic Pitch isn't installed. Run: pip install 'guitra[transcribe]'") from err
        self._predict = predict
        self._model = Model(ICASSP_2022_MODEL_PATH)  # load once, reuse for every chunk
        self.fmin = midi_to_hz(lowest_midi - 1)
        self.fmax = midi_to_hz(highest_midi + 1)
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

    def transcribe(self, samples: np.ndarray, sr: int, t0: float) -> list[NoteEvent]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk.wav"
            write_wav(path, samples, sr)
            _, _, note_events = self._predict(
                str(path),
                self._model,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_frequency=self.fmin,
                maximum_frequency=self.fmax,
            )
        # Basic Pitch note events: (start_s, end_s, midi_pitch, amplitude, pitch_bends)
        return [NoteEvent(int(p), t0 + float(s), t0 + float(e), float(a)) for s, e, p, a, *_ in note_events]


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Reads a 16-bit PCM WAV (mono or stereo → mono float32)."""
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return (data.reshape(-1, ch).mean(axis=1) if ch > 1 else data), sr
