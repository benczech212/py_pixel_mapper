#!/usr/bin/env python3
"""
Interactive camera previewer:
- Enumerates video capture devices
- Shows live preview
- Switch with LEFT/RIGHT arrows
- Displays index and device name (when available)
- Quit with 'q'

Dependencies:
    pip install opencv-python pygrabber  # pygrabber optional; improves names on Windows
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Optional

import cv2

# Optional: Windows device names via DirectShow
try:
    from pygrabber.dshow_graph import FilterGraph  # type: ignore
    _HAS_DSHOW = True
except Exception:
    _HAS_DSHOW = False


def get_windows_device_names() -> List[str]:
    if not _HAS_DSHOW:
        return []
    try:
        g = FilterGraph()
        return g.get_input_devices()
    except Exception:
        return []


def get_linux_device_names(max_devices: int) -> List[Tuple[int, str]]:
    names = []
    for i in range(max_devices):
        path = f"/sys/class/video4linux/video{i}/name"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    names.append((i, f.read().strip()))
            except Exception:
                pass
    return names


def probe_indices(max_devices: int, backend: Optional[int]) -> List[int]:
    """Return a list of indices that seem openable."""
    found = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            # Try to actually grab a frame (some backends report opened but don't deliver)
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()
    return found


def make_name_map(indices: List[int], platform: str, max_devices: int) -> dict:
    """Best-effort map from index -> human-readable name."""
    name_map = {}

    if platform == "win":
        names = get_windows_device_names()
        # DirectShow ordering often aligns with OpenCV indices, but not guaranteed.
        # We'll map by position where possible.
        for idx, cam_idx in enumerate(indices):
            if idx < len(names):
                name_map[cam_idx] = names[idx]
    elif platform == "linux":
        for cam_idx, nm in get_linux_device_names(max_devices):
            if cam_idx in indices:
                name_map[cam_idx] = nm

    # Fallback: label with just index if no name
    for cam_idx in indices:
        name_map.setdefault(cam_idx, f"Camera {cam_idx}")
    return name_map


def draw_label(frame, text: str):
    """Draw a readable label at top-left."""
    h, w = frame.shape[:2]
    pad, thickness = 10, 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = 8, 12 + th
    cv2.rectangle(frame, (x-6, y-th-6), (x+tw+6, y+6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def open_cap(index: int, width: int, height: int, fps: int, backend: Optional[int]):
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Preview each video device, switch with LEFT/RIGHT.")
    ap.add_argument("--max-devices", type=int, default=12, help="Probe up to this many indices (0..N-1).")
    ap.add_argument("--start", type=int, default=0, help="Start on this camera index if available.")
    ap.add_argument("--width", type=int, default=1280, help="Request width (0 = default).")
    ap.add_argument("--height", type=int, default=720, help="Request height (0 = default).")
    ap.add_argument("--fps", type=int, default=30, help="Request FPS (0 = default).")
    ap.add_argument("--backend", type=str, default="", choices=["", "msmf", "dshow", "avf", "v4l2"],
                    help="Force OpenCV backend (optional).")
    args = ap.parse_args()

    # Pick backend constant if requested
    backend_map = {
        "": None,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "avf": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2,
    }
    backend = backend_map[args.backend]

    platform = "win" if os.name == "nt" else ("linux" if sys.platform.startswith("linux") else "mac")

    indices = probe_indices(args.max_devices, backend)
    if not indices:
        print("No cameras found.")
        sys.exit(1)

    name_map = make_name_map(indices, platform, args.max_devices)

    # Choose starting camera
    if args.start in indices:
        cur_idx_pos = indices.index(args.start)
    else:
        cur_idx_pos = 0

    cap = open_cap(indices[cur_idx_pos], args.width, args.height, args.fps, backend)
    window = "Camera Preview (← / → to switch, q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_switch = 0.0
    SWITCH_COOLDOWN = 0.2  # seconds

    while True:
        ok, frame = cap.read()
        if not ok:
            # Try to reopen current
            cap.release()
            cap = open_cap(indices[cur_idx_pos], args.width, args.height, args.fps, backend)
            ok, frame = cap.read()
            if not ok:
                # If still not ok, move to next
                cur_idx_pos = (cur_idx_pos + 1) % len(indices)
                cap.release()
                cap = open_cap(indices[cur_idx_pos], args.width, args.height, args.fps, backend)
                continue

        idx = indices[cur_idx_pos]
        label = f"[{idx}] {name_map.get(idx, f'Camera {idx}')}"
        # Also show actual capture size/FPS (reported)
        rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rfps = cap.get(cv2.CAP_PROP_FPS)
        label += f"  {rw}x{rh} @ {int(rfps) if rfps else '?'}fps"

        draw_label(frame, label)
        cv2.imshow(window, frame)

        k = cv2.waitKey(1) & 0xFF
        now = time.time()

        if k == ord('q'):
            break
        elif k == 81 or k == 2424832:  # LEFT (Linux/Mac/Win codes)
            if now - last_switch > SWITCH_COOLDOWN:
                last_switch = now
                cur_idx_pos = (cur_idx_pos - 1) % len(indices)
                cap.release()
                cap = open_cap(indices[cur_idx_pos], args.width, args.height, args.fps, backend)
        elif k == 83 or k == 2555904:  # RIGHT
            if now - last_switch > SWITCH_COOLDOWN:
                last_switch = now
                cur_idx_pos = (cur_idx_pos + 1) % len(indices)
                cap.release()
                cap = open_cap(indices[cur_idx_pos], args.width, args.height, args.fps, backend)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
