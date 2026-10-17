"""Command line: `guitra run`, `guitra markers`, `guitra devices`."""

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .rig import DEFAULT_RIGS, Rig, load_rig


def _rig(args) -> Rig:
    return load_rig(args.rig) if args.rig else DEFAULT_RIGS[args.instrument]


def _make_transcriber(kind: str, rig: Rig):
    inst = rig.instrument
    lo, hi = inst.tuning[0], inst.tuning[-1] + inst.frets
    if kind == "basic-pitch":
        from .transcribe import BasicPitchTranscriber

        return BasicPitchTranscriber(lo, hi)
    if kind == "yin":
        from .transcribe import YinTranscriber

        return YinTranscriber(lo, hi)
    return None


def cmd_markers(args) -> None:
    import cv2

    from .markers import marker_sheet

    rig = _rig(args)
    cv2.imwrite(args.out, marker_sheet(rig))
    print(f"Wrote {args.out}. Print at 100% scale and check the marker size with a ruler.")
    for m in rig.markers:
        print(f"  id {m.id}: {m.size_mm:g} mm, centre at x={m.x_mm:g} mm from the nut, y={m.y_mm:g} mm from the centre line")


def cmd_devices(_args) -> None:
    from .audio import list_devices

    print(list_devices())


def cmd_run(args) -> None:
    import cv2

    from . import overlay
    from .app import LiveTranscription, Session
    from .hands import HandTracker
    from .transcribe import read_wav

    rig = _rig(args)
    session = Session(rig, HandTracker(Path(args.model)))
    if args.video:
        source = args.video
    else:
        try:
            source = int(args.camera)
        except ValueError:
            source = args.camera

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if args.video:
            sys.exit(f"Error: Couldn't open video file {source!r}. Check that the file exists and is a valid video.")
        sys.exit(
            f"Error: Couldn't open camera {source!r}.\n\n"
            f"Troubleshooting hints:\n"
            f"  - Check that your camera is plugged in and recognized by your system.\n"
            f"  - Ensure no other application (Zoom, Teams, Discord, browser) is using the camera.\n"
            f"  - Try another camera index (e.g. `--camera 1` or `--camera 2`).\n"
            f"  - On Windows, check Settings > Privacy & security > Camera to ensure camera access is allowed."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    transcriber = _make_transcriber(args.transcriber, rig)
    live = None
    file_events = []
    if transcriber and args.audio_file:
        samples, sr = read_wav(args.audio_file)
        print(f"Transcribing {args.audio_file} …")
        file_events = transcriber.transcribe(samples, sr, 0.0)
    elif transcriber and not args.video:
        from .audio import AudioInput

        audio = AudioInput(args.audio_device, channel=args.audio_channel)
        audio.start()
        window = 2.0 if args.transcriber == "basic-pitch" else 0.6
        live = LiveTranscription(audio, transcriber, window_s=window, hop_s=0.25)
        live.start()

    shown: list[str] = []
    frame_idx, start, last_fps_t, fps_count, cam_fps = 0, time.monotonic(), time.monotonic(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.video:  # recorded: use the video's own clock so audio-file notes line up
                t = frame_idx / fps
            else:
                t = time.monotonic()
            heard = live.events() if live else file_events
            result = session.step(frame, t, heard)
            for a in result.alerts:
                print(f"[{t - (0 if args.video else start):7.1f}s] {a.kind:13s} {a.message}")
                shown.append(a.message)
            frame_idx += 1
            fps_count += 1
            if time.monotonic() - last_fps_t >= 1:
                cam_fps, fps_count, last_fps_t = fps_count / (time.monotonic() - last_fps_t), 0, time.monotonic()
            if not args.no_window:
                pose = result.pose
                status = (
                    f"{rig.instrument.name} | markers {list(pose.marker_ids) if pose else '-'}"
                    f"{' (held)' if pose and pose.age_frames else ''} | {cam_fps:4.1f} fps"
                )
                img = overlay.draw(frame, rig.instrument, pose, session.neck.last_detections,
                                   result.hand, result.ringing, shown, status)
                cv2.imshow("guitra", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if live:
            live.stop()
        session.tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    kinds = {}
    for a in session.history:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    print("Session summary:", ", ".join(f"{k} ×{v}" for k, v in kinds.items()) or "no alerts")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="guitra", description="Practice coach for guitar and bass.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def rig_args(sp):
        sp.add_argument("--rig", help="TOML file describing your instrument and marker positions")
        sp.add_argument("--instrument", choices=sorted(DEFAULT_RIGS), default="guitar",
                        help="use the default rig for this instrument (ignored with --rig)")

    r = sub.add_parser("run", help="start practising with live feedback")
    rig_args(r)
    r.add_argument("--camera", default="0", help="camera index (default 0)")
    r.add_argument("--video", help="analyse a recorded video instead of the camera")
    r.add_argument("--audio-file", help="16-bit WAV recorded alongside --video")
    r.add_argument("--audio-device", default=None, help="input device index or name (see `guitra devices`)")
    r.add_argument("--audio-channel", type=int, default=0, help="input channel your amp/DI is on")
    r.add_argument("--transcriber", choices=["basic-pitch", "yin", "none"], default="yin",
                   help="basic-pitch handles chords (needs guitra[transcribe]); yin = single notes, built in")
    r.add_argument("--model", default="models/hand_landmarker.task", help="MediaPipe hand model path")
    r.add_argument("--no-window", action="store_true", help="print alerts only, no video window")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("markers", help="write a printable sheet of the ArUco markers for your rig")
    rig_args(m)
    m.add_argument("--out", default="markers.png")
    m.set_defaults(func=cmd_markers)

    d = sub.add_parser("devices", help="list audio input devices")
    d.set_defaults(func=cmd_devices)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
