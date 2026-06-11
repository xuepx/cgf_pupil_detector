"""
CGF Demo: generate a GIF showing confidence-guided fallback on LPW video.

The GIF visualizes per-frame CGF detection with:
  - Green ellipse/center: traditional detection (high confidence)
  - Red ellipse/center:   RITNet fallback (low confidence)
  - Confidence score bar and method indicator overlay

Usage:
    python demo.py --video LPW/1/1.avi --output demo.gif --frames 200
    python demo.py --lpw-dir LPW --participant 1 --vid-id 1 --output demo.gif
"""
import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from cgf_detector import CGFDetector


def load_lpw_annotations(txt_path):
    """Load LPW center annotations."""
    centers = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                centers.append((float(parts[0]), float(parts[1])))
    return centers


def render_frame(frame, result, gt_center=None, frame_idx=0):
    """
    Render a single frame with CGF detection overlay.

    Args:
        frame: BGR image (H, W, 3)
        result: dict from CGFDetector.detect()
        gt_center: (x, y) LPW annotation or None
        frame_idx: frame index for display

    Returns:
        rendered BGR image
    """
    canvas = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    # Color: green = traditional, red = RITNet fallback
    is_trad = result['method'] == 'traditional'
    color = (0, 220, 80) if is_trad else (60, 60, 255)  # BGR
    method_str = "Traditional" if is_trad else "RITNet Fallback"

    # Draw ellipse and center
    if result['found']:
        if result['ellipse'] is not None:
            cv2.ellipse(canvas, result['ellipse'], color, 2)
        if result['center'] is not None:
            cx, cy = int(result['center'][0]), int(result['center'][1])
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

    # Draw GT center
    if gt_center is not None:
        gx, gy = int(gt_center[0]), int(gt_center[1])
        cv2.circle(canvas, (gx, gy), 4, (255, 255, 0), 1)  # yellow

    # --- Info panel (top bar) ---
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    # Method label
    cv2.putText(canvas, f"#{frame_idx}  {method_str}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Confidence score
    conf = result['confidence']
    cv2.putText(canvas, f"conf={conf:.3f}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Confidence bar
    bar_x, bar_w = w - 160, 140
    cv2.rectangle(canvas, (bar_x, 10), (bar_x + bar_w, 22), (60, 60, 60), -1)
    fill_w = int(bar_w * conf)
    bar_color = (0, 200, 80) if conf >= 0.8 else (0, 200, 255) if conf >= 0.7 else (60, 60, 255)
    cv2.rectangle(canvas, (bar_x, 10), (bar_x + fill_w, 22), bar_color, -1)
    cv2.putText(canvas, f"tau=0.8", (bar_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Center error vs GT
    if result['found'] and gt_center is not None and result['center'] is not None:
        dx = result['center'][0] - gt_center[0]
        dy = result['center'][1] - gt_center[1]
        err = np.sqrt(dx**2 + dy**2)
        cv2.putText(canvas, f"err={err:.1f}px",
                    (w - 160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return canvas


def make_gif(frames_bgr, output_path, durations, fps=8):
    """Save list of BGR frames as GIF with per-frame durations.

    Args:
        frames_bgr: list of BGR images
        output_path: output GIF path
        durations: list of per-frame durations in ms (same length as frames_bgr)
        fps: fallback fps if durations not provided
    """
    from PIL import Image
    pil_frames = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=True
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"GIF saved: {output_path} ({len(pil_frames)} frames, {size_mb:.1f} MB)")


def run_demo(video_path, annot_path, output_path, max_frames=200,
             weights_path=None, threshold=0.8, gif_fps=8, downscale=1,
             trad_subsample=3, ritnet_linger=3):
    """Run CGF demo on a single LPW video and save GIF.

    Args:
        trad_subsample: keep 1 out of every N traditional frames (reduces size)
        ritnet_linger: duration multiplier for RITNet frames (makes fallback visible)
    """
    detector = CGFDetector(weights_path=weights_path, threshold=threshold)
    annotations = load_lpw_annotations(annot_path) if annot_path and os.path.exists(annot_path) else []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_count = min(total, max_frames) if max_frames else total
    print(f"Video: {video_path}  Frames: {total}, Processing: {process_count}")
    print(f"Threshold: {threshold}, Weights: {weights_path or 'default'}")
    print(f"Trad subsample: keep 1/{trad_subsample}, RITNet linger: {ritnet_linger}x")

    rendered_frames = []
    frame_durations = []  # per-frame duration in ms
    base_duration = int(1000 / gif_fps)
    stats = {'traditional': 0, 'RITNet': 0, 'not_found': 0}
    trad_count = 0  # counter for subsampling traditional frames

    for frame_idx in tqdm(range(process_count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        result = detector.detect(gray)
        stats[result['method']] = stats.get(result['method'], 0) + 1
        if not result['found']:
            stats['not_found'] += 1

        is_ritnet = result['method'] != 'traditional'

        # Subsample traditional frames: keep every Nth
        if not is_ritnet:
            trad_count += 1
            if trad_count % trad_subsample != 1 and trad_subsample > 1:
                continue

        gt_center = annotations[frame_idx] if frame_idx < len(annotations) else None
        rendered = render_frame(frame, result, gt_center, frame_idx)

        if downscale > 1:
            h, w = rendered.shape[:2]
            rendered = cv2.resize(rendered, (w // downscale, h // downscale))

        rendered_frames.append(rendered)
        # RITNet frames linger longer
        duration = base_duration * ritnet_linger if is_ritnet else base_duration
        frame_durations.append(duration)

    cap.release()

    # Print stats
    total_processed = sum(stats.values()) - stats['not_found']
    trad_pct = stats['traditional'] / max(total_processed, 1) * 100
    ritnet_pct = stats['RITNet'] / max(total_processed, 1) * 100
    print(f"\n--- CGF Statistics ---")
    print(f"  Traditional: {stats['traditional']} ({trad_pct:.1f}%)")
    print(f"  RITNet fallback: {stats['RITNet']} ({ritnet_pct:.1f}%)")
    print(f"  Not found: {stats['not_found']}")
    print(f"  GIF frames kept: {len(rendered_frames)} (from {process_count} processed)")

    make_gif(rendered_frames, output_path, durations=frame_durations, fps=gif_fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGF Demo GIF Generator')
    parser.add_argument('--video', type=str, help='Path to LPW .avi video')
    parser.add_argument('--annot', type=str, help='Path to LPW .txt annotation')
    parser.add_argument('--lpw-dir', type=str, default=None,
                        help='LPW dataset root (e.g. LPW/)')
    parser.add_argument('--participant', type=int, default=1,
                        help='Participant ID (used with --lpw-dir)')
    parser.add_argument('--vid-id', type=int, default=1,
                        help='Video ID (used with --lpw-dir)')
    parser.add_argument('--weights', type=str, default=None,
                        help='RITNet weights path (.pkl)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='CGF confidence threshold (default: 0.7)')
    parser.add_argument('--frames', type=int, default=150,
                        help='Max frames to process')
    parser.add_argument('--gif-fps', type=int, default=8,
                        help='Output GIF frame rate')
    parser.add_argument('--downscale', type=int, default=2,
                        help='Downscale factor for smaller GIF')
    parser.add_argument('--trad-subsample', type=int, default=3,
                        help='Keep 1/N traditional frames (default: 3, reduces size)')
    parser.add_argument('--ritnet-linger', type=int, default=3,
                        help='Duration multiplier for RITNet frames (default: 3x)')
    parser.add_argument('--output', type=str, default='demo.gif',
                        help='Output GIF path')
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        annot_path = args.annot or video_path.replace('.avi', '.txt')
    elif args.lpw_dir:
        video_path = os.path.join(args.lpw_dir, str(args.participant), f"{args.vid_id}.avi")
        annot_path = os.path.join(args.lpw_dir, str(args.participant), f"{args.vid_id}.txt")
    else:
        print("ERROR: Provide --video or --lpw-dir")
        sys.exit(1)

    run_demo(
        video_path=video_path, annot_path=annot_path, output_path=args.output,
        max_frames=args.frames, weights_path=args.weights, threshold=args.threshold,
        gif_fps=args.gif_fps, downscale=args.downscale,
        trad_subsample=args.trad_subsample, ritnet_linger=args.ritnet_linger
    )
