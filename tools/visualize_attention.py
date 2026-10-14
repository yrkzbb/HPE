#!/usr/bin/env python3
"""Create a Figure-4-style JTA attention comparison.

The same trained checkpoint is evaluated twice on the same temporal crop:
once with mask-aware feature-to-joint attention disabled and once enabled.
This is an inference-time intervention visualization, not a replacement for
the paper's separately trained mask ablation.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from models import TAR_ViTPose
from inference import list_frame_files, load_cfg, preprocess_frame, set_seed


# Internally this implementation uses the 17-slot PoseTrack-COCO ordering
# (cfg.MODEL.NUM_JOINTS == 17), even though PoseTrack reports 15 joints.
# In this internal ordering: right_shoulder=6, right_elbow=8, right_wrist=10.
JOINTS = [(8, "right elbow"), (10, "right wrist"), (6, "right shoulder")]


def parse_args():
    p = argparse.ArgumentParser("Visualize TAR-ViTPose JTA attention")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-w", "--weights", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--detector", default="checkpoints/yolo11n.pt")
    p.add_argument("--output", default="output/figure4_demo/figure4_attention.jpg")
    p.add_argument("--frame_index", type=int, default=-1,
                   help="center-frame index; -1 selects the middle frame")
    p.add_argument("--person_index", type=int, default=0,
                   help="person index after sorting detections by box area")
    p.add_argument("--layer", type=int, default=-1,
                   help="JTA layer whose attention is visualized")
    p.add_argument("--source_offset", type=int, default=0,
                   help="source-frame position in the temporal window; 0 is t-T")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def normalize_map(attn):
    attn = attn.astype(np.float32)
    attn -= attn.min()
    maximum = attn.max()
    if maximum > 0:
        attn /= maximum
    return attn


def overlay_attention(image_rgb, attn):
    h, w = image_rgb.shape[:2]
    attn = cv2.resize(normalize_map(attn), (w, h), interpolation=cv2.INTER_CUBIC)
    attn = np.clip(attn, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(attn * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.55, heatmap, 0.45, 0)


def collect_attention(model, inp, use_mask, layer, num_frames, feature_h, feature_w):
    model.use_mask = use_mask
    with torch.no_grad():
        model(inp)
    weights = model.last_attn_maps[layer]
    return weights.view(
        weights.shape[0], weights.shape[1], num_frames, feature_h, feature_w
    ).cpu().numpy()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg.SEED)
    if torch.cuda.is_available() and args.gpu >= 0:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    files = list_frame_files(args.frames_dir)
    center_idx = len(files) // 2 if args.frame_index < 0 else args.frame_index
    half = cfg.WINDOWS_SIZE // 2
    indices = [min(max(center_idx + delta, 0), len(files) - 1)
               for delta in range(-half, half + 1)]
    frames = [cv2.imread(files[i]) for i in indices]
    if any(frame is None for frame in frames):
        raise RuntimeError("Failed to read one or more temporal frames")

    detector = YOLO(args.detector)
    detections = detector.predict(frames[half], verbose=False, classes=0,
                                  conf=args.conf)[0].boxes
    if len(detections) == 0:
        raise RuntimeError("No person detected in the selected center frame")
    boxes = sorted((box.xyxy[0].cpu().numpy() for box in detections),
                   key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    if args.person_index >= len(boxes):
        raise IndexError(f"person_index={args.person_index}, detections={len(boxes)}")
    x1, y1, x2, y2 = boxes[args.person_index]
    image_h, image_w = frames[half].shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = (x2 - x1) * cfg.DATASET.BBOX_ENLARGE_FACTOR
    height = (y2 - y1) * cfg.DATASET.BBOX_ENLARGE_FACTOR
    x1 = int(max(cx - width / 2, 0))
    y1 = int(max(cy - height / 2, 0))
    x2 = int(min(cx + width / 2, image_w))
    y2 = int(min(cy + height / 2, image_h))

    crops_bgr = [frame[y1:y2, x1:x2] for frame in frames]
    inp = torch.stack([preprocess_frame(crop, cfg.MODEL.IMAGE_SIZE)
                       for crop in crops_bgr]).unsqueeze(0).to(device)

    model = TAR_ViTPose(cfg, phase="test", device=device)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    no_mask = collect_attention(model, inp, False, args.layer, cfg.WINDOWS_SIZE,
                                model.f_h, model.f_w)
    with_mask = collect_attention(model, inp, True, args.layer, cfg.WINDOWS_SIZE,
                                  model.f_h, model.f_w)

    source_rgb = cv2.cvtColor(crops_bgr[args.source_offset], cv2.COLOR_BGR2RGB)
    current_rgb = cv2.cvtColor(crops_bgr[half], cv2.COLOR_BGR2RGB)
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title(r"Neighbor frame $X_i(t-T)$")
    axes[1, 0].imshow(current_rgb)
    axes[1, 0].set_title(r"Current frame $X_i(t)$")

    for column, (joint_index, joint_name) in enumerate(JOINTS, start=1):
        axes[0, column].imshow(overlay_attention(
            source_rgb, no_mask[0, joint_index, args.source_offset]))
        axes[0, column].set_title(f"without mask: {joint_name}")
        axes[1, column].imshow(overlay_attention(
            source_rgb, with_mask[0, joint_index, args.source_offset]))
        axes[1, column].set_title(f"with mask: {joint_name}")

    for ax in axes.flat:
        ax.axis("off")
    figure.suptitle("JTA joint-query attention on the neighboring frame", fontsize=16)
    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved attention visualization to {args.output}")
    print(f"Center frame: {files[center_idx]}")
    print(f"Person box: {(x1, y1, x2, y2)}")


if __name__ == "__main__":
    main()
