<p align="center">
  <img src="assets/banner.svg" alt="Infinite Void" width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10--3.12-black?style=for-the-badge&logo=python&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-computer%20vision-black?style=for-the-badge&logo=opencv&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-hand%20tracking-black?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-black?style=for-the-badge">
</p>

<h3 align="center">Cross your fingers at your webcam and the world turns black and white,<br>and a black hole opens up.</h3>

---

## About

**Infinite Void** is a small real-time computer vision project written in Python. It watches your webcam, and when it sees you make the crossed-fingers hand sign (index and middle finger crossed, the other fingers curled), the scene transforms:

- the picture fades from color to high-contrast **black and white**
- a **black hole** opens in the center of the frame
- a glowing **whitish accretion ring** and spiral arms swirl around it
- the video **bends around the hole** like gravitational lensing
- stars twinkle in the dark

Drop the sign and reality comes back.

> This is a fan-made project inspired by the "Infinite Void" domain expansion. It is not affiliated with or endorsed by the creators or publishers of *Jujutsu Kaisen*.

## Features

- Real-time hand tracking with MediaPipe's `HandLandmarker`
- Rule-based detection of the crossed-fingers sign (no training needed)
- Smooth fade in and out so the effect never pops
- Swirl, vignette, glow and star effects done with NumPy and OpenCV
- A debug mode that draws the hand skeleton and prints why the sign was or wasn't detected

## Requirements

- Python **3.10 to 3.12** (64-bit)
- A webcam
- Windows, macOS or Linux

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
python -m pip install opencv-python mediapipe numpy
```

## Usage

```bash
python gojo.py
```

A window called **Infinite Void** opens with your mirrored webcam feed.

1. Raise your hand so the camera can see it clearly, palm toward the camera.
2. Cross your **index and middle fingers**, and curl your **ring finger and pinky**.
3. Hold the sign for about a second.
4. Watch the void open.

The first run downloads the hand-tracking model (`hand_landmarker.task`, about 8 MB) automatically. After that it works offline.

### Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `d` | Toggle debug mode (hand skeleton, detection status, terminal logs) |

## How it works

1. **Capture:** OpenCV reads frames from the webcam and mirrors them.
2. **Hand landmarks:** MediaPipe's `HandLandmarker` returns 21 landmarks per hand.
3. **Sign detection:** the script checks that the index and middle fingers are extended, the ring and pinky are curled, and that the two fingertips have moved toward or past each other compared with their knuckles, which is what "crossed" looks like in 2D.
4. **Smoothing:** the sign must be held for a few frames, then an intensity value `t` eases from 0 to 1 (and back).
5. **Rendering:** for each frame, `t` controls a lensing swirl (`cv2.remap`), the grayscale blend, a vignette, a glowing ring and spiral arms, stars, and the black disc itself.

## Configuration

Edit the constants at the top of `gojo.py`:

| Setting | Default | What it does |
| --- | --- | --- |
| `CAM_INDEX` | `0` | Which camera to use (try `1` if you have several) |
| `HOLD_FRAMES` | `4` | Frames the sign must be held before triggering |
| `FADE_IN` / `FADE_OUT` | `0.07` / `0.10` | How fast the void opens and closes |
| `HOLE_RADIUS_FRAC` | `0.16` | Black hole size relative to frame height |

If the sign triggers too easily or not at all, adjust the `0.45` threshold inside `is_crossed_fingers`. Lower is stricter, higher is more forgiving.

## Troubleshooting

<details>
<summary><b>"Could not open the webcam"</b></summary>

Close other apps using the camera (Zoom, Teams, browser tabs) and try `CAM_INDEX = 1`. On Windows, check **Settings → Privacy & security → Camera** and allow desktop apps to access it.
</details>

<details>
<summary><b><code>pip</code> is not recognized (Windows)</b></summary>

Python isn't on your PATH. Reinstall it from python.org with **"Add python.exe to PATH"** ticked, reopen your terminal, and use `python -m pip install ...`.
</details>

<details>
<summary><b><code>module 'mediapipe' has no attribute 'solutions'</code></b></summary>

Newer MediaPipe versions removed the old `solutions` API. This project already uses the newer Tasks API, so make sure you're running the latest `gojo.py`.
</details>

<details>
<summary><b>The sign isn't detected</b></summary>

Press `d` for debug mode and watch the terminal. It prints which finger checks pass or fail and the crossing ratio. Use good lighting, keep your whole hand in frame, and face your palm toward the camera.
</details>

## Project structure

```
.
├── assets/
│   └── banner.svg          # README banner
├── gojo.py                 # main script
├── hand_landmarker.task    # downloaded automatically on first run
└── README.md
```

## Ideas for the future

- Make the black hole follow your hand
- Segment the person so only the background goes black and white
- Add a sound effect when the void opens
- Support more hand signs and effects

## Credits

- [OpenCV](https://opencv.org/) for video capture and image processing
- [MediaPipe](https://ai.google.dev/edge/mediapipe) for hand landmark detection
- [NumPy](https://numpy.org/) for the effect math

## License

Released under the MIT License. Add a `LICENSE` file to your repository to make it official.
