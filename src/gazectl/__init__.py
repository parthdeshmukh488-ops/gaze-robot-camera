"""Hands-free control of a robot-held camera with eye tracking.

The package is deliberately split so that the parts carrying the algorithms
have no dependency on a camera, a GUI toolkit or a physics simulator:

    geometry     screen geometry and the pixel-to-visual-angle conversion
    filters      One Euro smoothing, I-VT fixation detection
    calibration  per-user gaze-to-screen mapping, accuracy and precision
    selection    dwell and look-and-confirm techniques
    robot        the safety layer between gaze and a moving camera
    study        SUS and NASA-TLX scoring
    tracking     MediaPipe adapter        (optional, needs mediapipe)
    ui           PySide6 front end        (optional, needs PySide6)

Everything above the last two runs on numpy alone, which is why the test suite
covers it on a machine with no webcam attached.
"""

from .geometry import ScreenGeometry

__version__ = "0.1.0"
__all__ = ["ScreenGeometry"]
