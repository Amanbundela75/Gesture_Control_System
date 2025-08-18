# ✋ AI Gesture Control System 🤖  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#prerequisites)
[![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-success)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-%23ff69b4.svg)](#-contributing)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgray)](#-platform-notes)

Control your computer using nothing but your hands.  
This Python-powered project uses your webcam to detect hand landmarks in real time and map specific gestures to system actions like:
- Navigating presentations
- Moving a virtual pointer
- Adjusting system volume
- Taking instant screenshots  
…and more (extensible)!

---

## 🧭 Table of Contents

1. [Features](#-features)
2. [System Architecture](#-system-architecture)
3. [Gestures & Mappings](#-gestures--mappings)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
6. [Running the App](#-run-the-application)
7. [Usage Guide](#-usage-guide)
8. [Configuration & Customization](#-configuration--customization)
9. [Performance Tips](#-performance--stability-tips)
10. [Troubleshooting](#-troubleshooting)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)
14. [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- 🔍 **Real-Time Hand Landmark Tracking**  
  Tracks 21 hand keypoints using MediaPipe with millisecond-level responsiveness.
- 🖥 **Presentation Navigation**  
  Open palm + swipe → next / previous slide.
- 🖱 **Pointer / Cursor Mode**  
  Index finger isolation controls a virtual cursor (mapped to screen coordinates).
- 🔊 **Volume Control (Fist Depth)**  
  Closed fist distance to camera scales to volume level.
- 📸 **Instant Screenshot**  
  Peace ✌️ gesture triggers a saved screenshot.
- 🧠 **Gesture State Overlay**  
  On-screen HUD: active mode, frame rate, detected gesture.
- 🧩 **Easily Extensible**  
  Add custom gestures and map them to keystrokes or system actions.
- 🛡 **Non-Intrusive**  
  Does not require special hardware—just a standard webcam.

---

## 🧬 System Architecture

```
Webcam Feed
     │
     ▼
 OpenCV Frame Capture
     │
     ▼
 MediaPipe Hand Detector
     │  (21 landmarks)
     ▼
 Gesture Analyzer (logic & heuristics)
     │
     ├─► Action Mapper (PyAutoGUI / OS APIs)
     │
     └─► Visual Feedback (HUD overlay)
     │
     ▼
 Display Window (Real-time output)
```

Core pipeline phases:
1. Frame acquisition  
2. Landmark detection  
3. Gesture classification (finger states, distances, angles)  
4. Action execution (keyboard/mouse/system)  
5. Feedback rendering  

---

## ✋ Gestures & Mappings

| Gesture | Description | Default Action |
|--------|-------------|----------------|
| Open Palm (5 fingers) + Horizontal Swipe | Directional motion threshold | Slide change (← / →) |
| Single Index Finger | Finger tip tracked | Cursor / laser pointer |
| Closed Fist | Z-depth change toward/away camera | Volume down / volume up |
| Peace ✌️ (Index + Middle) | Fingers spaced | Capture screenshot |
| (Placeholder) Thumb Up | (Optional) | Could map to Play/Pause |
| (Placeholder) OK Sign | (Optional) | Could confirm dialogs |

> You can expand gestures by adding rules for finger combinations, angles between landmarks, or temporal sequences.

---

## ✅ Prerequisites

- Python 3.9+ (3.10+ recommended)
- Webcam
- A desktop environment (not headless)
- Permissions to control system audio (for volume features)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Amanbundela75/Gesture-Control-System.git
cd Gesture-Control-System
```

(Adjust the URL if your repository name differs.)

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a requirements file yet, generate one after manually installing packages:
```bash
pip install opencv-python mediapipe pyautogui numpy
pip freeze > requirements.txt
```

---

## ▶ Run the Application

```bash
python gesture_control.py
```

Optional arguments (if implemented):
```
--camera 0          # Select a different camera index
--no-volume         # Disable volume control logic
--debug             # Show raw landmark coordinates
--smoothing 4       # Adjust cursor smoothing factor
```

---

## 🧑‍💻 Usage Guide

1. Launch the script; a window displaying the camera feed will open.  
2. Ensure your hand is within a well-lit area (avoid overexposure).  
3. Perform gestures:
   - Slide navigation: Open palm + swipe left/right
   - Cursor mode: Show only index finger
   - Volume control: Make a fist and move it closer/farther
   - Screenshot: Show peace sign ✌️ (saved as screenshot_YYYYMMDD_HHMMSS.png)
4. Press Esc (or Ctrl + C in terminal) to exit.

---

## ⚙ Configuration & Customization

You can customize behavior by editing (examples):

```python
# Example gesture thresholds (hypothetical snippet)
SWIPE_MIN_DISTANCE = 120         # Pixels across frame
CURSOR_SMOOTHING   = 0.25        # Lower = more precise, higher = smoother
FIST_DEPTH_MIN     = 0.05        # Normalized z-range
FIST_DEPTH_MAX     = 0.25
SCREENSHOT_DIR     = "screenshots"
```

Add a new gesture mapping (conceptual outline):
```python
if is_thumb_up(landmarks):
    perform_media_play_pause()
```

Ideas for expansion:
- Multi-hand coordination
- Gesture sequences (e.g., fist → open → index = mode toggle)
- UI overlay toggles
- Config file (JSON / YAML) for dynamic gesture/action mapping

---

## 🚀 Performance & Stability Tips

- Reduce frame size (e.g., 640x480) for smoother FPS.
- Use a dedicated lighting source to minimize detection jitter.
- Apply exponential smoothing to cursor coordinates:
  `smoothed = alpha * new + (1 - alpha) * previous`
- Avoid running in Remote Desktop sessions—some OS APIs block input control.
- On laptops, disable auto-exposure if flickering occurs (OpenCV camera properties).

---

## 🧪 Testing Ideas

| Test Case | Expected Result |
|-----------|-----------------|
| Hand partially out of frame | No false gesture triggers |
| Two gestures quickly | Only one action executed |
| Low light conditions | Landmarks degrade gracefully |
| Rapid swipes | Slide changes max once per swipe |
| Fist near camera | Max volume threshold clamped |

You can automate parts of logic with synthetic landmark arrays in unit tests.

---

## 🛠 Platform Notes

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Cursor Control | ✅ | ✅ | ✅ |
| Volume Control | ✅ (PyAutoGUI / ctypes) | ⚠ (May require AppleScript) | ⚠ (Depends on pactl / amixer) |
| Screenshots | ✅ | ✅ | ✅ |

You may implement OS-specific volume controllers:
- macOS: `osascript -e "set volume output volume X"`
- Linux: `pactl set-sink-volume @DEFAULT_SINK@ 50%`

---

## ❗ Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Window opens but no gestures | Wrong camera index | Try `--camera 1` |
| High latency | Large frame size | Downscale frames before processing |
| Cursor jitter | Minor landmark noise | Increase smoothing factor |
| Volume not changing | OS not supported | Disable volume feature or add OS hook |
| Screenshot not saved | No write permission | Run in writable directory |

Enable debug overlays or log prints for diagnosing coordinate drift.

---

## 🗺 Roadmap

- [ ] Configurable gesture-to-action JSON
- [ ] Multi-hand simultaneous actions
- [ ] GUI settings panel
- [ ] Temporal gesture recognition (LSTM / transformer)
- [ ] Dark mode optimized segmentation
- [ ] Plugin architecture for add-on actions
- [ ] In-app calibration wizard
- [ ] Export analytics (gesture usage frequency)

Feel free to propose additions via Issues / Discussions.

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork the repository  
2. Create a feature branch: `git checkout -b feat/your-feature`  
3. Commit changes: `git commit -m "Add feature: description"`  
4. Push branch: `git push origin feat/your-feature`  
5. Open a Pull Request (clearly describe motivation & approach)

Suggested quality checks:
- Run `flake8` or `ruff` (if configured)
- Test on at least one alternate lighting condition
- Document new gestures in this README

---

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [MediaPipe Hands](https://developers.google.com/mediapipe) for robust hand landmark detection.
- OpenCV community for foundational computer vision tooling.
- Inspiration from various HCI (Human-Computer Interaction) prototypes.

---

## 🗨 Feedback

Have an idea or found a bug? Open an Issue or start a Discussion.  
If this project helps you, consider starring ⭐ the repository!

---

Happy gesturing! ✨  
