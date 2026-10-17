import pytest
from unittest.mock import patch, MagicMock

from guitra import __version__
from guitra.cli import main, cmd_run


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert f"guitra {__version__}" in captured.out


def test_cmd_run_camera_failure():
    args = MagicMock()
    args.rig = None
    args.instrument = "guitar"
    args.model = "models/hand_landmarker.task"
    args.video = None
    args.camera = "99"
    args.transcriber = "none"

    with patch("cv2.VideoCapture") as mock_cap, \
         patch("guitra.app.Session"), \
         patch("guitra.hands.HandTracker"):
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        with pytest.raises(SystemExit) as exc:
            cmd_run(args)
        assert "Troubleshooting hints" in str(exc.value)


def test_cmd_run_video_failure():
    args = MagicMock()
    args.rig = None
    args.instrument = "guitar"
    args.model = "models/hand_landmarker.task"
    args.video = "nonexistent_video.mp4"
    args.camera = "0"
    args.transcriber = "none"

    with patch("cv2.VideoCapture") as mock_cap, \
         patch("guitra.app.Session"), \
         patch("guitra.hands.HandTracker"):
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        with pytest.raises(SystemExit) as exc:
            cmd_run(args)
        assert "Couldn't open video file" in str(exc.value)
