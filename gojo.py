"""
Infinite Void - webcam hand-sign effect (new MediaPipe Tasks API)
-----------------------------------------------------------------
Cross your index + middle finger (keep ring + pinky curled) and hold it.
The scene fades to black & white and a swirling black hole opens up.

Setup:
    python -m pip install opencv-python mediapipe numpy

Run:
    python gojo.py

Keys:  q = quit    d = toggle debug info
The hand model (~8 MB) is downloaded automatically the first time.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------- settings
CAM_INDEX = 0
HOLD_FRAMES = 4
FADE_IN = 0.07
FADE_OUT = 0.10
HOLE_RADIUS_FRAC = 0.16

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


# ------------------------------------------------------------ sign detection
def _d(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def is_crossed_fingers(lm, debug=False):
    """True if index + middle are extended and crossed, ring + pinky curled."""
    wrist = lm[0]

    def extended(tip, pip):
        return _d(lm[tip], wrist) > _d(lm[pip], wrist) * 1.05

    def curled(tip, pip):
        return _d(lm[tip], wrist) < _d(lm[pip], wrist) * 1.05

    index_up = extended(8, 6)
    middle_up = extended(12, 10)
    ring_down = curled(16, 14)
    pinky_down = curled(20, 18)

    bx = lm[9].x - lm[5].x
    by = lm[9].y - lm[5].y
    base_sep = np.hypot(bx, by)
    if base_sep < 1e-6:
        return False
    ux, uy = bx / base_sep, by / base_sep
    tip_sep = (lm[12].x - lm[8].x) * ux + (lm[12].y - lm[8].y) * uy
    ratio = tip_sep / base_sep

    if debug:
        print(f"idx={index_up} mid={middle_up} ring_down={ring_down} "
              f"pinky_down={pinky_down} ratio={ratio:.2f}")

    if not (index_up and middle_up and ring_down and pinky_down):
        return False
    return ratio < 0.45


# ------------------------------------------------------------- void effect
class Void:
    def __init__(self, w, h):
        self.w, self.h = w, h
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.cx, self.cy = w / 2, h / 2
        dx, dy = xs - self.cx, ys - self.cy
        self.r = np.sqrt(dx * dx + dy * dy)
        self.theta = np.arctan2(dy, dx)

        rng = np.random.default_rng(7)
        stars = np.zeros((h, w), np.float32)
        n = 350
        stars[rng.integers(0, h, n), rng.integers(0, w, n)] = rng.uniform(0.4, 1.0, n)
        self.stars = cv2.GaussianBlur(stars, (0, 0), 1.2) * 6

    def apply(self, frame, t, now):
        h, w = self.h, self.w
        R = h * HOLE_RADIUS_FRAC * (0.2 + 0.8 * t)

        # gravitational-lensing swirl
        swirl_zone = R * 4.5
        falloff = np.clip(1 - self.r / swirl_zone, 0, 1) ** 2
        a = self.theta - falloff * 3.2 * t
        map_x = (self.cx + self.r * np.cos(a)).astype(np.float32)
        map_y = (self.cy + self.r * np.sin(a)).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # black & white with punchy contrast
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gray = np.clip((gray - 0.5) * 1.6 + 0.45, 0, 1) * 0.8
        gray3 = np.dstack([gray] * 3)
        out = (warped.astype(np.float32) / 255) * (1 - t) + gray3 * t

        # vignette
        out *= (1 - 0.55 * t * np.clip(self.r / (w * 0.6), 0, 1))[..., None]

        # whitish accretion ring + spiral arms + halo
        ring = np.exp(-((self.r - R * 1.25) / (R * 0.28)) ** 2)
        spiral = 0.5 + 0.5 * np.sin(3 * self.theta - self.r * 0.045 + now * 5)
        arms = spiral * np.exp(-np.clip(self.r - R, 0, None) / (R * 1.6))
        halo = np.exp(-np.clip(self.r - R, 0, None) / (R * 0.9)) * 0.55
        glow = np.clip((ring * 1.2 + arms * 0.55 + halo) * t, 0, 1)[..., None]
        out = out + glow * np.array([1.0, 0.97, 0.93], np.float32)

        # stars
        out += (self.stars * (self.r > R)).astype(np.float32)[..., None] * t * 0.6

        # the black hole
        hole = np.clip((R - self.r) / (R * 0.12), 0, 1)[..., None]
        out = out * (1 - hole)

        # flash as it opens
        if 0.01 < t < 0.35:
            out += (0.35 - t) * 0.8

        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


# ------------------------------------------------------------------- main
def main():
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try CAM_INDEX = 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Webcam opened but returned no frames.")
    h, w = frame.shape[:2]
    void = Void(w, h)

    t, held, debug = 0.0, 0, False
    start = time.time()
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        sign = False
        for lm in result.hand_landmarks:
            if is_crossed_fingers(lm, debug):
                sign = True
            if debug:
                draw_hand(frame, lm)

        held = held + 1 if sign else 0
        active = held >= HOLD_FRAMES
        t = min(1.0, t + FADE_IN) if active else max(0.0, t - FADE_OUT)

        out = void.apply(frame, t, time.time()) if t > 0 else frame

        if t > 0.6:
            cv2.putText(out, "Domain Expansion: Infinite Void", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
        if debug:
            cv2.putText(out, f"sign={sign} t={t:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Infinite Void", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()