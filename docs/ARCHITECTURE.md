# guitra architecture

## Pipeline

Every camera frame goes through `Session.step()` (`guitra/app.py`), which has no I/O of its own,
so the same code runs on a live camera, a recorded video, or test fixtures.

```
frame ─► NeckLocator.update()  (neck.py)      ArUco markers → homography pixels ↔ mm
      ─► HandTracker.detect()  (hands.py)     MediaPipe → 21 landmarks per hand
      ─► fretting_hand()       (mapping.py)   fingertips → neck mm → (string, fret)
audio ─► Transcriber           (transcribe.py) Basic Pitch / YIN → NoteEvents
      ─► Coach.update()        (coach.py)     hand + heard notes → alerts
      ─► overlay.draw()        (overlay.py)   debug view
```

Live audio is captured on its own thread (`audio.py`, sounddevice ring buffer). A background
`LiveTranscription` thread re-transcribes the latest window every 0.25 s. All timestamps use one
clock (`time.monotonic()` live, the video's own time for recordings), so heard notes line up with
camera frames.

## Coordinates

**Neck coordinates** (`fretboard.py`) are millimetres:

- `x`: from the nut towards the body. Fret *n* is at `L · (1 − 2^(−n/12))`.
- `y`: from the neck's centre line. Strings fan out linearly from the nut spread to the bridge spread.

`locate(x, y)` returns the nearest string, the fret space the point is in, how deep into that
fret space it is (0 = just past the previous fret wire, 1 = on the wire you're pressing), and
whether it's on the neck at all.

Strings are indexed from the lowest-pitched (0 = low E), and `Instrument.string_label()` gives
players' numbering ("string 6 (E2)").

## Finding the fretboard (OpenCV ArUco)

The rig file (`rig.py`, `examples/*.toml`) lists each marker's id, size and centre in neck
coordinates. Markers are stuck upright as seen from the front (top edge towards the low strings),
which fixes where each of the 4 corners sits on the neck. `NeckLocator` detects markers with
`cv2.aruco.ArucoDetector` (sub-pixel corners) and fits a homography to all corner pairs:

- Two or more markers spread along the neck: reliable (green grid).
- One marker: works but extrapolates far from it (grey grid, flagged unreliable).
- Fits are smoothed between frames, and the last good fit is held for ~15 frames when markers are
  briefly hidden.

The homography assumes the markers and fretboard are roughly one plane. Headstock angle and
fretboard height add a few millimetres of error, small next to fret spacing (17–36 mm).

## Fretting hand (MediaPipe)

`HandTracker` runs MediaPipe's HandLandmarker in video mode. The fretting hand is the one with
the most fingertips on the fretboard. That's more robust than MediaPipe's left/right label, which
depends on camera mirroring. Each fingertip (index 8, middle 12, ring 16, pinky 20) is
projected into neck coordinates. A finger counts as *on a string* when it's on the neck and within
45% of a string spacing of that string's centre.

2D tracking can't tell whether a finger is pressing or hovering. The coach only judges a finger
once the audio confirms its note (or, for muted strings, confirms a chord was strummed across it).

## Transcription

- **BasicPitchTranscriber**: Spotify Basic Pitch, polyphonic. The model loads once; each window is
  written to a temporary WAV and run through `basic_pitch.inference.predict`. Optional install.
- **YinTranscriber**: YIN pitch detection (FFT-based difference function, cumulative mean
  normalisation, parabolic interpolation). Monophonic but tiny and fast. Good for riffs, scales and
  bass lines, and it handles low E on bass (41 Hz).

Both return `NoteEvent(midi, start, end, amplitude)` on the session clock.

## Coach rules (`coach.py`)

For each string with fingers on it, the finger at the highest fret is the one that sounds.
`ringing` is the set of notes heard in the last 0.6 s.

| Rule | Logic |
|---|---|
| muted-string | ≥ 2 notes ringing (a chord), the fretted note lies inside the strummed range, and it isn't heard (exact pitch; chords often double notes in octaves). Mentions an adjacent finger. |
| wrong-fret | A heard note isn't explained by any fretted or open string, but is 1–2 semitones from a fretted note. |
| far-from-fret | The finger's note is heard, but the fingertip is in the back 40% of its fret space. |
| lazy-pinky | Index on fret *n*, ring finger sounding fret ≥ *n+3*, pinky not on that fret. |
| flying-pinky | The other fingers are on the neck, and the pinky is > 25 mm outside the neck edge, behind the nut, or lifted towards the camera (MediaPipe z) by > 0.45 × hand size, for 1 s. |

Every alert must persist (0.35 s, or 1 s for the pinky) and has a 5 s cooldown per key.

## Tests

`tests/synth.py` renders camera frames with real ArUco markers warped into perspective, fake hands
at known fret positions, and synthetic notes, so the whole pipeline is tested without a camera,
guitar or amp. Basic Pitch is tested against its API shape with a stand-in module, since the real
package needs a large ML runtime.

## Roadmap

1. Tune thresholds on real recordings (`guitra run --video --audio-file` makes this repeatable).
2. Exercises: load a scale or chord chart, and check the expected notes and fingering, not just consistency.
3. Thumb position (over the neck vs behind it) from the thumb landmark.
4. Picking hand: alternate picking and string muting from the second hand.
5. Lower-latency chord transcription (smaller windows, or Basic Pitch's TF.js/ONNX runtime directly).
