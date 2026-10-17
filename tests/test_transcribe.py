import sys
import types

import numpy as np
import pytest

from guitra.notes import midi_to_hz, name_to_midi
from guitra.transcribe import BasicPitchTranscriber, YinTranscriber, read_wav, write_wav, yin_pitch

SR = 22050


def tone(midi, seconds=0.5, saw=False):
    t = np.arange(int(SR * seconds)) / SR
    f = midi_to_hz(midi)
    if saw:  # bright, harmonic-rich like a real string
        return (0.3 * (2 * ((t * f) % 1) - 1)).astype(np.float32)
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.mark.parametrize("note", ["E1", "A1", "E2", "G3", "E4", "B4"])
def test_yin_finds_the_pitch(note):
    midi = name_to_midi(note)
    hz = yin_pitch(tone(midi, saw=True)[:2048], SR, 35, 1000)
    cents = 1200 * np.log2(hz / midi_to_hz(midi))
    assert abs(cents) < 15, (note, hz)


def test_yin_transcriber_splits_notes():
    audio = np.concatenate([tone(name_to_midi("E2"), 0.4, saw=True), tone(name_to_midi("G2"), 0.4, saw=True)])
    events = YinTranscriber(name_to_midi("E2"), name_to_midi("E5")).transcribe(audio, SR, t0=10.0)
    assert [e.midi for e in events] == [name_to_midi("E2"), name_to_midi("G2")]
    assert events[0].start == pytest.approx(10.0, abs=0.05)
    assert events[1].start == pytest.approx(10.4, abs=0.1)


def test_silence_is_not_a_note():
    assert YinTranscriber(40, 88).transcribe(np.zeros(SR, np.float32), SR, 0.0) == []


def test_wav_round_trip(tmp_path):
    audio = tone(60, 0.1)
    write_wav(tmp_path / "a.wav", audio, SR)
    back, sr = read_wav(tmp_path / "a.wav")
    assert sr == SR and np.allclose(back, audio, atol=1e-3)


def test_basic_pitch_adapter(monkeypatch):
    """Basic Pitch itself is an optional, heavy install; check our adapter against its API shape."""
    calls = {}

    def fake_predict(path, model, **kw):
        calls["path"], calls["model"], calls["kw"] = path, model, kw
        read_wav(path)  # the chunk must be a readable WAV
        return None, None, [(0.1, 0.6, 52, 0.7, []), (0.1, 0.6, 59, 0.5, [])]

    pkg = types.ModuleType("basic_pitch")
    pkg.ICASSP_2022_MODEL_PATH = "model.onnx"
    inference = types.ModuleType("basic_pitch.inference")
    inference.Model = lambda path: f"model<{path}>"
    inference.predict = fake_predict
    monkeypatch.setitem(sys.modules, "basic_pitch", pkg)
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", inference)

    events = BasicPitchTranscriber(40, 88).transcribe(tone(52), SR, t0=5.0)
    assert [(e.midi, e.start, e.end) for e in events] == [(52, 5.1, 5.6), (59, 5.1, 5.6)]
    assert calls["model"] == "model<model.onnx>"
    assert calls["kw"]["minimum_frequency"] < midi_to_hz(40)
