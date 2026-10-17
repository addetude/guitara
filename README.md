# guitra

A practice coach for electric guitar and bass. When you're teaching yourself, nobody tells you
what you're doing wrong. guitra watches your fretting hand, listens to your amp, and tells you
when something's off, like a muted string or a lazy pinky.

## How it works

```
 camera ──► MediaPipe hand tracker ──► fingertip pixels ─┐
        └─► OpenCV ArUco markers ──► fretboard homography ┴─► string + fret per finger ─┐
                                                                                         ├─► coach ─► alerts
 amp / audio interface ──► note transcription (Spotify Basic Pitch, or built-in YIN) ───┘
```

1. **Hands**: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
   finds 21 landmarks per hand in each camera frame.
2. **Fretboard**: printed [ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
   markers stuck to the headstock and body are detected with OpenCV. Their known positions give a
   homography from image pixels to millimetres on the neck, and fret positions follow the
   12th-root-of-two rule. Together, every fingertip maps to a **string and fret**.
3. **Sound**: audio from your amp/interface is transcribed into notes.
   [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) handles chords. A small built-in
   YIN pitch detector covers single notes without the heavy install.
4. **Coach**: compares what your fingers are doing with what was actually heard:

| Alert | When it fires |
|---|---|
| **muted string** | You strum a chord, a finger is fretting a string inside it, but that note isn't ringing. It also names the neighbouring finger that's probably leaning on it. |
| **wrong fret** | A note is heard a semitone or two away from what your fingers show. |
| **far from fret** | A finger is sounding a note from far behind the fret wire (buzz, extra pressure). |
| **lazy pinky** | Your index is on fret *n* and the ring finger stretches to *n+3* instead of using the pinky (one finger per fret). |
| **flying pinky** | Your pinky drifts far off the neck (or lifts up towards the camera) while the other fingers play. |

Alerts have to persist briefly before firing and won't repeat for a few seconds.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[audio,dev]"          # add ,transcribe for Basic Pitch (large install)
```

1. **Print markers**: `guitra markers --rig examples/guitar.toml --out markers.png`. Print at
   100% scale and check the size with a ruler.
2. **Stick them on**: one on the headstock, two on the body either side of the neck. Put them
   upright as seen from the front, top edge towards the low strings.
3. **Measure**: copy `examples/guitar.toml` (or `bass.toml`) and enter each marker's centre:
   `x_mm` from the nut along the neck, `y_mm` from the neck's centre line (negative = low-string side).
   Also set your scale length.
4. **Play**:

```bash
guitra devices                                   # find your audio interface
guitra run --rig my-guitar.toml --audio-device 2 # live camera + amp
guitra run --rig my-guitar.toml --transcriber basic-pitch --audio-device 2   # chords
guitra run --rig my-bass.toml --video take.mp4 --audio-file take.wav          # review a recording
```

The window shows the detected markers, the fretboard grid, each finger's string/fret and alerts.
Press `q` to quit. A summary of alerts is printed at the end.

Point the camera at the front of the neck so both the headstock and body markers are visible.

## Status

Early prototype. The pipeline runs end to end and is unit-tested with synthetic camera frames,
hands and audio. It still needs:

- Testing and threshold tuning with real guitars, cameras and amps
- Basic Pitch runs on 2 s windows every 0.25 s, so expect roughly a second of delay for chord alerts
- Thumb position, picking-hand feedback and timing/rhythm checks aren't built yet

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how each piece works.

## Development

```bash
pytest          # 33 tests: fretboard geometry, marker detection, mapping, coach rules, transcription
```
