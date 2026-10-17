"""guitra: a practice coach for electric guitar and bass.

Camera: MediaPipe finds your fretting hand; ArUco markers on the neck (found with OpenCV)
tell us where the fretboard is, so each fingertip maps to a string and fret.
Audio: the signal from your amp/interface is transcribed into notes.
Coach: comparing what your hand is doing with what was heard produces alerts
(muted strings, wrong frets, fingers far from the fret wire, a lazy or flying pinky).
"""

__version__ = "0.1.0"
