#!/usr/bin/env python3
"""
Interactive Pixelblaze mapper:
- Pick BL/BR on baseline, live perpendicular & top preview, confirm/reselect
- Pilot scan first 10 pixels with progress bar + overlay; tweak dwell/threshold and rerun or continue
- Full scan with progress bar + live overlay; retry misses; per-pixel annotated images (optional)
- Bounding-box -> target aspect (default 16:9) -> 0..1 map export for Pixelblaze
- Interactive final preview with hover tooltips
- Channel-order verification (needs PB pattern export "testAll")

Requires:
  pip install websocket-client opencv-python pyyaml tqdm
"""

import argparse, json, os, platform, sys, time, datetime
import cv2, numpy as np, yaml
from websocket import create_connection
from tqdm import tqdm

# ---------------- Pixelblaze helpers ----------------
def set_vars(ws, **kwargs):
    ws.send(json.dumps({"setVars": kwargs}))

# ---------------- Vision helpers ----------------
def find_all_centroids(baseline_bgr, lit_bgr, thresh=25, min_area=8, blur=5):
    base_gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    lit_gray  = cv2.cvtColor(lit_bgr,      cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(lit_gray, base_gray)
    if blur >= 3 and blur % 2 == 1:
        diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    if cnts:
        for c in cnts:
            area = int(cv2.contourArea(c))
            if area < min_area: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
            blob_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            bright = float(cv2.mean(diff, mask=blob_mask)[0])
            centroids.append((float(cx), float(cy), area, bright))
    return centroids, mask

def choose_centroid_lowest_x(centroids):
    if not centroids: return None, None
    centroids.sort(key=lambda t: (t[0], -t[3], -t[2]))  # lowest X; break ties by brightness/area
    return centroids[0][0], centroids[0][1]

def draw_label(frame, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (10, 10), (10 + tw + 14, 10 + th + 18), (0,0,0), -1)
    cv2.putText(frame, text, (16, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def render_points_overlay(background_img, points, radius=5, label_every=0, bbox=None):
    img = background_img.copy()
    if bbox is not None:
        (xmin, ymin, xmax, ymax) = bbox
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
    for i, pt in enumerate(points):
        if pt is None: continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius, (0, 0, 255), -1)
        if label_every and (i % label_every == 0):
            cv2.putText(img, str(i), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return img

# -------- Baseline capture (with optional AE kick) --------
def grab_nonblack_baseline(cap, ws, args):
    if args.kick_exposure:
        set_vars(ws, background=float(args.kick_level))
        time.sleep(args.kick_ms / 1000.0)
        set_vars(ws, background=0.0)
    if args.baseline_warmup_sec > 0: time.sleep(args.baseline_warmup_sec)
    for _ in range(max(0, args.baseline_frames)): cap.read()
    deadline = time.time() + args.baseline_timeout_sec
    frame_ok = None
    while True:
        ok, frame = cap.read()
        if ok:
            mean_val = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            if mean_val >= args.baseline_min_mean or time.time() >= deadline:
                frame_ok = frame; break
        if time.time() >= deadline:
            frame_ok = frame if ok else None; break
    if frame_ok is None: raise RuntimeError("Failed to capture any baseline frame.")
    return frame_ok

# ------------- Geometry helpers -------------
def compute_bbox(points):
    xs = [p[0] for p in points if p is not None]
    ys = [p[1] for p in points if p is not None]
    if not xs or not ys: return None
    return (min(xs), min(ys), max(xs), max(ys))

def normalize_to_aspect(points, bbox, target_aspect=(16,9)):
    if bbox is None:
        return [None if p is None else (0.5,0.5) for p in points], None
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin; h = ymax - ymin
    if w <= 0 or h <= 0:
        return [None if p is None else (0.5,0.5) for p in points], None
    target_ratio = float(target_aspect[0]) / float(target_aspect[1])
    box_ratio = w / h
    if box_ratio > target_ratio:
        new_h = w / target_ratio; pad = (new_h - h)/2
        x0, x1 = xmin, xmax; y0, y1 = ymin - pad, ymax + pad
    else:
        new_w = h * target_ratio; pad = (new_w - w)/2
        x0, x1 = xmin - pad, xmax + pad; y0, y1 = ymin, ymax
    out = []
    for p in points:
        if p is None: out.append(None); continue
        x, y = p
        x01 = (x - x0) / (x1 - x0)
        y01 = 1.0 - (y - y0) / (y1 - y0)  # flip Y for PB
        out.append((max(0.0,min(1.0,x01)), max(0.0,min(1.0,y01))))
    return out, (x0, y0, x1, y1)

# ------------- Interactive axis picker (BL, BR, TOP with live preview) -------------
def pick_quad_homography(baseline_img):
    """
    Click order: BL, BR, TOP(anywhere). We build a parallelogram:
      P0=BL, P1=BR, P2=TR=TOP+(BR-BL), P3=TOP
    Then compute H: image(x,y) -> (u,v) in unit square (0..1), y up.
    Returns: (BL, BR, TOP, H, quad_pts)
    """
    img = baseline_img.copy()
    win = "Baseline: BL, BR, then TOP (Y confirm, R reselect)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    picks = []   # BL, BR, TOP
    BL = BR = TOP = None
    TR_preview = None
    quad_preview = None

    def draw_scene(mouse=None):
        canvas = img.copy()
        # draw existing picks
        if len(picks) >= 1:
            cv2.circle(canvas, (int(picks[0][0]), int(picks[0][1])), 6, (0,255,255), -1)
            cv2.putText(canvas, "BL", (int(picks[0][0])+8, int(picks[0][1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        if len(picks) >= 2:
            cv2.circle(canvas, (int(picks[1][0]), int(picks[1][1])), 6, (0,255,255), -1)
            cv2.putText(canvas, "BR", (int(picks[1][0])+8, int(picks[1][1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            # draw baseline
            cv2.line(canvas, (int(picks[0][0]), int(picks[0][1])),
                             (int(picks[1][0]), int(picks[1][1])), (0,255,0), 2)

            # live TOP preview at mouse: free anywhere
            if mouse is not None and len(picks) == 2:
                BLv = np.array(picks[0], dtype=np.float64)
                BRv = np.array(picks[1], dtype=np.float64)
                mv  = np.array(mouse,   dtype=np.float64)
                TRv = mv + (BRv - BLv)   # opposite corner to form a parallelogram
                nonlocal TR_preview, quad_preview
                TR_preview = (float(TRv[0]), float(TRv[1]))
                TOP_preview = (float(mv[0]), float(mv[1]))
                quad_preview = [picks[0], picks[1], TR_preview, TOP_preview]
                # draw quad
                q = quad_preview
                for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                    cv2.line(canvas, (int(q[a][0]), int(q[a][1])),
                                     (int(q[b][0]), int(q[b][1])), (255,0,0), 1)
                cv2.circle(canvas, (int(TOP_preview[0]), int(TOP_preview[1])), 7, (0,200,255), 2)

        # instructions
        if len(picks) == 0: msg = "Click BOTTOM-LEFT"
        elif len(picks) == 1: msg = "Click BOTTOM-RIGHT"
        elif len(picks) == 2: msg = "Move mouse to preview TOP, click to place. Y=confirm, R=reset"
        else: msg = "Y=confirm, R=reset"
        draw_label(canvas, msg)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            draw_scene((x,y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(picks) < 2:
                picks.append((float(x), float(y)))
                draw_scene((x,y))
            elif len(picks) == 2:
                # place TOP wherever clicked
                picks.append((float(x), float(y)))
                draw_scene((x,y))

    cv2.setMouseCallback(win, on_mouse)
    draw_scene()

    H = None
    quad = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('r'), ord('R')):
            picks.clear()
            draw_scene()
            continue
        if len(picks) == 3 and key in (ord('y'), ord('Y')):
            BL, BR, TOP = picks
            BLv = np.array(BL, dtype=np.float64)
            BRv = np.array(BR, dtype=np.float64)
            TOPv= np.array(TOP,dtype=np.float64)
            TRv = TOPv + (BRv - BLv)
            quad = np.array([BLv, BRv, TRv, TOPv], dtype=np.float32)

            # dst: unit square with y up: (0,0)->bottom-left, so in image coords that’s (0,1)
            dst = np.array([[0.0,1.0],
                            [1.0,1.0],
                            [1.0,0.0],
                            [0.0,0.0]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(quad, dst)  # maps image (x,y,1) -> (u,v,1)

            # quick confirmation overlay
            canvas = img.copy()
            for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                cv2.line(canvas, (int(quad[a][0]), int(quad[a][1])),
                                 (int(quad[b][0]), int(quad[b][1])), (0,255,255), 2)
            draw_label(canvas, "Quad chosen. Y=confirm, R=reselect")
            cv2.imshow(win, canvas)

            # wait immediate confirm
            key2 = cv2.waitKey(0) & 0xFF
            if key2 in (ord('y'), ord('Y')):
                cv2.destroyWindow(win)
                return BL, BR, TOP, H, quad.tolist()
            else:
                picks.clear()
                draw_scene()
def map_points_with_homography(points_xy, H):
    """
    points_xy: list of None or (x_img, y_img)
    H: 3x3 matrix mapping image -> unit square (u,v) with y up
    Returns list of None or (u,v) in 0..1
    """
    out = []
    for p in points_xy:
        if p is None:
            out.append(None); continue
        x, y = p
        vec = np.array([x, y, 1.0], dtype=np.float64)
        uvw = H.dot(vec)
        if abs(uvw[2]) < 1e-9:
            out.append(None); continue
        u = float(uvw[0]/uvw[2])
        v = float(uvw[1]/uvw[2])
        out.append((max(0.0, min(1.0, u)), max(0.0, min(1.0, v))))
    return out

# ------------- Channel-order verification (global) -------------
def measure_global_color(cap, baseline, settle_frames=4, blur=3):
    for _ in range(settle_frames): cap.read()
    ok, frame = cap.read()
    if not ok: return None, None
    base_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, base_gray)
    if blur >= 3 and blur % 2 == 1:
        diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    return frame, diff

def sample_window(img, x, y, r=3):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x)-r), min(w, int(x)+r+1)
    y0, y1 = max(0, int(y)-r), min(h, int(y)+r+1)
    if x0 >= x1 or y0 >= y1: return 0.0
    return float(np.mean(img[y0:y1, x0:x1]))

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Scan all pixels, live preview, retry misses, and save YAML.")
    # Defaults you requested
    ap.add_argument("--ip", default="192.168.4.78", help="Pixelblaze IP (e.g., 192.168.4.78)")
    ap.add_argument("--count", type=int, default=1215, help="Logical pixel count (e.g., 1215)")
    ap.add_argument("--yaml-out", default="mappings_all.yaml", help="Output YAML filename")

    # Camera defaults for Windows
    ap.add_argument("--camera", type=int, default=2, help="Camera index (default 2)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)

    # Scan timing/vision
    ap.add_argument("--background", type=float, default=0.0, help="Pattern background while scanning (0..1)")
    ap.add_argument("--dwell", type=float, default=0.10, help="Seconds to wait after lighting before capture")
    ap.add_argument("--settle-frames", type=int, default=2, help="Frames to discard before capture")
    ap.add_argument("--thresh", type=int, default=25, help="Diff threshold (0..255)")
    ap.add_argument("--min-area", type=int, default=8, help="Minimum contour area")
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel (odd; 0 disables)")
    ap.add_argument("--set-total", action="store_true", help="Set pattern totalPixels to --count")
    ap.add_argument("--save-images", action="store_true", help="Save per-pixel lit/mask/annot images")

    # Live preview
    ap.add_argument("--preview", dest="preview", action="store_true", help="Show live preview overlay while scanning")
    ap.add_argument("--no-preview", dest="preview", action="store_false", help="Disable live preview")
    ap.set_defaults(preview=True)
    ap.add_argument("--label-every", type=int, default=0, help="Label every Nth point in preview (0=none)")
    ap.add_argument("--point-radius", type=int, default=5, help="Point radius in preview dots")

    # Retry pass
    ap.add_argument("--retry-iters", type=int, default=1, help="How many retry rounds for missing pixels")
    ap.add_argument("--retry-dwell", type=float, default=0.30, help="Dwell during retry passes (s)")
    ap.add_argument("--retry-thresh", type=int, default=20, help="Threshold during retries (more sensitive)")
    ap.add_argument("--retry-min-area", type=int, default=8, help="Min area during retries")

    # Baseline robustness
    ap.add_argument("--baseline-warmup-sec", type=float, default=1.0, help="Extra time to let camera AE settle")
    ap.add_argument("--baseline-frames", type=int, default=30, help="Frames to discard before baseline grab")
    ap.add_argument("--baseline-min-mean", type=float, default=5.0, help="Min grayscale mean to accept baseline")
    ap.add_argument("--baseline-timeout-sec", type=float, default=5.0, help="Give up waiting after this long")
    ap.add_argument("--kick-exposure", dest="kick_exposure", action="store_true",                help="Briefly light background to wake AE before baseline")
    ap.add_argument("--no-kick-exposure", dest="kick_exposure", action="store_false",                help="Disable AE kick")
    ap.set_defaults(kick_exposure=True)
    ap.add_argument("--kick-level", type=float, default=0.08, help="Background level during AE kick (0..1)")
    ap.add_argument("--kick-ms", type=int, default=300, help="Kick duration in ms")

    # Aspect normalization/export
    ap.add_argument("--aspect-w", type=int, default=16, help="Target aspect width (default 16)")
    ap.add_argument("--aspect-h", type=int, default=9,  help="Target aspect height (default 9)")
    ap.add_argument("--export-json", default="pixel_map_2d.json", help="Pixelblaze 0..1 2D map JSON output")

    # Per-run folder
    ap.add_argument("--run-prefix", default="mapping_run", help="Folder prefix; timestamp appended")




    args = ap.parse_args()

    # Per-run folder
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.run_prefix}_{ts}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Open camera (DirectShow on Windows) ---
    backend = cv2.CAP_DSHOW if platform.system().lower().startswith("win") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    time.sleep(0.3)
    for _ in range(4): cap.read()
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}", file=sys.stderr); sys.exit(1)

    # --- WebSocket / Pixelblaze init ---
    ws = create_connection(f"ws://{args.ip}:81", timeout=5)
    init_payload = {"useHSV": 0, "background": float(args.background)}
    if args.set_total: init_payload["totalPixels"] = int(args.count)
    set_vars(ws, **init_payload); set_vars(ws, R=0.0, G=0.0, B=0.0, testAll=0)

    # --- Baseline capture ---
    baseline = grab_nonblack_baseline(cap, ws, args)
    cv2.imwrite(os.path.join(run_dir, "baseline.jpg"), baseline)

    # --- Axis picking + confirmation ---
    BL, BR, TOP, H, quad_pts = pick_quad_homography(baseline)

    # derive axes (normalized) and center
    BLv = np.array(BL); BRv = np.array(BR); Cv = 0.5*(BLv+BRv)
    ex = (BRv - BLv); L = np.linalg.norm(ex); ex = ex / max(1e-9, L)
    ey = np.array([-ex[1], ex[0]])
    # orient ey toward TOP
    if np.dot(np.array(TOP)-Cv, ey) < 0: ey = -ey

    # --- Scan helpers ---
    def scan_range(indices, dwell, thresh, min_area, label, save_suffix=""):
        points = [None]*len(indices)
        overlay_win = "Live preview (all points so far)"
        if args.preview: cv2.namedWindow(overlay_win, cv2.WINDOW_NORMAL)
        for idx_k, i in enumerate(tqdm(indices, desc=label, ncols=80)):
            set_vars(ws, pixelIndex=int(i)); set_vars(ws, R=1.0, G=0.0, B=0.0)
            time.sleep(dwell)
            for _ in range(max(1, args.settle_frames)): cap.read()
            ok, lit = cap.read()
            set_vars(ws, R=0.0, G=0.0, B=0.0)

            cx = cy = None; mask = None
            if ok:
                cents, mask = find_all_centroids(baseline, lit, thresh=thresh, min_area=min_area, blur=args.blur)
                cx, cy = choose_centroid_lowest_x(cents)
                if args.save_images:
                    cv2.imwrite(os.path.join(run_dir, f"pix_{i:05d}_lit{save_suffix}.jpg"), lit)
                    cv2.imwrite(os.path.join(run_dir, f"pix_{i:05d}_mask{save_suffix}.png"), mask)
                    annot = lit.copy()
                    if cx is not None and cy is not None:
                        cv2.circle(annot, (int(round(cx)), int(round(cy))), 8, (0,255,255), 2)
                        draw_label(annot, f"#{i} {label} ({cx:.1f},{cy:.1f})")
                    else:
                        draw_label(annot, f"#{i} {label} MISS")
                    cv2.imwrite(os.path.join(run_dir, f"pix_{i:05d}_annot{save_suffix}.jpg"), annot)
            points[idx_k] = None if cx is None else (cx, cy)

            if args.preview:
                # build overlay of all known points so far (pilot/full separately)
                sofar = [None]*args.count
                for k, idx_real in enumerate(indices[:idx_k+1]):
                    sofar[idx_real] = points[k]
                overlay = render_points_overlay(baseline, sofar, radius=args.point_radius, label_every=args.label_every)
                draw_label(overlay, f"{label}: {idx_k+1}/{len(indices)}  (global idx {i})")
                cv2.imshow(overlay_win, overlay); cv2.waitKey(1)
        return points

    # ---------------- Pilot: first 10 ----------------
    pilot_n = min(10, args.count)
    pilot_indices = list(range(pilot_n))
    pilot_pts = scan_range(pilot_indices, args.dwell, args.thresh, args.min_area, label="Pilot")

    # Pilot preview + control loop
    def draw_pilot_preview():
        sofar = [None]*args.count
        for k, i in enumerate(pilot_indices): sofar[i] = pilot_pts[k]
        canvas = render_points_overlay(baseline, sofar, radius=args.point_radius, label_every=args.label_every)
        draw_label(canvas, f"Pilot done. dwell={args.dwell:.2f}s  thresh={args.thresh}  "
                           f"[ / ] dwell  - / = thresh  R=rerun  C=continue  Q=quit")
        cv2.imshow("Pilot preview (tweak or continue)", canvas)

    draw_pilot_preview()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('['),):  args.dwell = max(0.03, args.dwell - 0.02); draw_pilot_preview()
        elif key in (ord(']'),): args.dwell = args.dwell + 0.02; draw_pilot_preview()
        elif key in (ord('-'),): args.thresh = max(1, args.thresh - 2); draw_pilot_preview()
        elif key in (ord('='),): args.thresh = args.thresh + 2; draw_pilot_preview()
        elif key in (ord('r'), ord('R')):
            pilot_pts = scan_range(pilot_indices, args.dwell, args.thresh, args.min_area, label="Pilot_rerun", save_suffix="_r")
            draw_pilot_preview()
        elif key in (ord('c'), ord('C')):
            cv2.destroyWindow("Pilot preview (tweak or continue)")
            break
        elif key in (ord('q'), ord('Q')):
            print("Aborted after pilot."); cv2.destroyAllWindows(); ws.close(); cap.release(); sys.exit(0)

    # ---------------- Full scan (remaining) ----------------
    pixels_xy = [None]*args.count
    for k, i in enumerate(pilot_indices):
        pixels_xy[i] = pilot_pts[k]

    remaining = [i for i in range(pilot_n, args.count)]
    if remaining:
        full_pts = scan_range(remaining, args.dwell, args.thresh, args.min_area, label="Scan")
        for k, i in enumerate(remaining):
            pixels_xy[i] = full_pts[k]

    # Retry pass for misses
    miss_idx = [i for i, v in enumerate(pixels_xy) if v is None]
    if miss_idx and args.retry_iters > 0:
        for r in range(args.retry_iters):
            if not miss_idx: break
            # scan misses with more forgiving settings
            misses_pts = scan_range(miss_idx, args.retry_dwell, args.retry_thresh, args.retry_min_area,
                                    label=f"Retry{r+1}", save_suffix=f"_r{r+1}")
            new_miss = []
            for k, i in enumerate(miss_idx):
                if misses_pts[k] is None: new_miss.append(i)
                else: pixels_xy[i] = misses_pts[k]
            miss_idx = new_miss

    # ------------- Bounding boxes & aspect normalization -------------
    bbox_tight = compute_bbox(pixels_xy)
    norm_points = map_points_with_homography(pixels_xy, H)

    # Save PB 0..1 JSON
    pb_array = [[0.0, 0.0] if v is None else [round(v[0],6), round(v[1],6)] for v in norm_points]
    with open(os.path.join(run_dir, args.export_json), "w", encoding="utf-8") as f:
        json.dump(pb_array, f, separators=(",", ":"))

    # Save YAML
    results = {
        "metadata": {
            "ip": args.ip, "count": int(args.count),
            "frame_size": [int(args.width), int(args.height)], "fps_req": int(args.fps),
            "background": float(args.background), "dwell_s": float(args.dwell),
            "thresh": int(args.thresh), "min_area": int(args.min_area), "blur": int(args.blur),
            "camera_index": int(args.camera), "backend": "CAP_DSHOW" if backend==cv2.CAP_DSHOW else "default",
            "retry": {"iters": int(args.retry_iters), "dwell_s": float(args.retry_dwell),
                      "thresh": int(args.retry_thresh), "min_area": int(args.retry_min_area)},
            "aspect_target": [args.aspect_w, args.aspect_h],
            "bbox_tight": None if bbox_tight is None else [float(x) for x in bbox_tight],
            "run_dir": run_dir,
            "axes": {"BL": {"x": float(BL[0]), "y": float(BL[1])},
                     "BR": {"x": float(BR[0]), "y": float(BR[1])},
                     "TOP": {"x": float(TOP[0]), "y": float(TOP[1])}},
        },
        "pixels": []
    }
    results["metadata"]["homography"] = {
        "quad_src": quad_pts,         # [[BL],[BR],[TR],[TOP]]
        "dst_square": [[0,1],[1,1],[1,0],[0,0]],
        "H": H.tolist()
    }
    for i, pt in enumerate(pixels_xy):
        entry = {"index": i, "centroid": None if pt is None else {"x": float(pt[0]), "y": float(pt[1])}}
        if norm_points[i] is not None:
            entry["normalized_xy01"] = {"x": float(norm_points[i][0]), "y": float(norm_points[i][1])}
        results["pixels"].append(entry)
    with open(os.path.join(run_dir, args.yaml_out), "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False)

       # ------------- Previews (homography-based) -------------
    # 1) Tight bbox on raw detections
    overlay1 = render_points_overlay(baseline, pixels_xy, radius=args.point_radius, label_every=args.label_every,
                                     bbox=bbox_tight)
    draw_label(overlay1, f"Tight bbox | missing: {sum(1 for v in pixels_xy if v is None)}")
    cv2.imwrite(os.path.join(run_dir, "preview_bbox_tight.jpg"), overlay1)

    # 2) Quad overlay (your BL, BR, TR, TOP) on baseline
    quad_img = baseline.copy()
    q = quad_pts  # [[BL],[BR],[TR],[TOP]]
    for a, b in [(0,1), (1,2), (2,3), (3,0)]:
        xa, ya = int(q[a][0]), int(q[a][1])
        xb, yb = int(q[b][0]), int(q[b][1])
        cv2.line(quad_img, (xa, ya), (xb, yb), (0, 255, 255), 2)
    draw_label(quad_img, "Selected quad (BL→BR→TR→TOP)")
    cv2.imwrite(os.path.join(run_dir, "preview_quad.jpg"), quad_img)

    # 3) Interactive tooltip preview: project normalized (u,v) back to image with H^{-1}
    Hinv = np.linalg.inv(H)
    pts_img = []
    for v in norm_points:
        if v is None:
            pts_img.append(None)
            continue
        u, vv = v
        vec = np.array([u, vv, 1.0], dtype=np.float64)
        xyw = Hinv.dot(vec)
        x = float(xyw[0] / xyw[2])
        y = float(xyw[1] / xyw[2])
        pts_img.append((x, y))

    # build the visualization base
    vis = quad_img.copy()
    for p in pts_img:
        if p is None:
            continue
        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

    win = "Final preview (hover for index, 0..1)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, mx, my, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        img = vis.copy()
        # nearest point within ~10px
        best = None
        best_d2 = 10 * 10
        for i, p in enumerate(pts_img):
            if p is None:
                continue
            dx = p[0] - mx
            dy = p[1] - my
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = i
        if best is not None and norm_points[best] is not None:
            x01, y01 = norm_points[best]
            txt = f"#{best}  ({x01:.3f}, {y01:.3f})"
            cv2.circle(img, (int(pts_img[best][0]), int(pts_img[best][1])), 7, (0, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx, by = mx + 12, my - 10
            cv2.rectangle(img, (bx, by - th - 8), (bx + tw + 10, by + 8), (0, 0, 0), -1)
            cv2.putText(img, txt, (bx + 5, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, vis)
    cv2.waitKey(1)


    # ------------- Channel order verification -------------
    set_vars(ws, testAll=1)
    # Red
    set_vars(ws, R=1.0, G=0.0, B=0.0); time.sleep(0.15)
    frameR, diffR = measure_global_color(cap, baseline)
    # Green
    set_vars(ws, R=0.0, G=1.0, B=0.0); time.sleep(0.15)
    frameG, diffG = measure_global_color(cap, baseline)
    # Blue
    set_vars(ws, R=0.0, G=0.0, B=1.0); time.sleep(0.15)
    frameB, diffB = measure_global_color(cap, baseline)
    set_vars(ws, R=0.0, G=0.0, B=0.0, testAll=0)

    channel_orders = []
    if frameR is not None and frameG is not None and frameB is not None:
        cv2.imwrite(os.path.join(run_dir, "global_R.jpg"), frameR)
        cv2.imwrite(os.path.join(run_dir, "global_G.jpg"), frameG)
        cv2.imwrite(os.path.join(run_dir, "global_B.jpg"), frameB)
        for i, pt in enumerate(pixels_xy):
            if pt is None: channel_orders.append("unknown"); continue
            x, y = pt
            r_val = sample_window(diffR, x, y, r=3) if diffR is not None else 0.0
            g_val = sample_window(diffG, x, y, r=3) if diffG is not None else 0.0
            b_val = sample_window(diffB, x, y, r=3) if diffB is not None else 0.0
            triplet = [('R', r_val), ('G', g_val), ('B', b_val)]
            triplet.sort(key=lambda t: t[1], reverse=True)
            if triplet[0][1] < 2.0 or triplet[0][1] < 1.2 * (triplet[1][1]+1e-6):
                order = "uncertain"
            else:
                order = "".join([t[0] for t in triplet])
            channel_orders.append(order)
    else:
        channel_orders = ["unknown"] * args.count

    # add to YAML + write again
    for i, entry in enumerate(results["pixels"]):
        entry["channel_order"] = channel_orders[i]
    with open(os.path.join(run_dir, args.yaml_out), "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    # Save key artifacts path summary
    print("\nArtifacts saved in:", run_dir)
    print(" - Baseline:", os.path.join(run_dir, "baseline.jpg"))
    print(" - YAML:    ", os.path.join(run_dir, args.yaml_out))
    print(" - PB map:  ", os.path.join(run_dir, args.export_json))
    print(" - Previews:", "preview_bbox_tight.jpg, preview_bbox_fit.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    try: ws.close()
    except Exception: pass
    cap.release()

if __name__ == "__main__":
    main()
