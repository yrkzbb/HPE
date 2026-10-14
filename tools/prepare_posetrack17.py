#!/usr/bin/env python3
"""Convert the original PoseTrack 2017 MAT annotations for TAR-ViTPose.

The generated COCO files contain the metadata expected by this repository
(`nframes` and `frame_id`).  Per-video JSON files are also emitted for the
bundled poseval evaluator.  Existing images are referenced in place; they are
not copied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


KEYPOINT_NAMES = [
    "nose", "head_bottom", "head_top", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]

# PoseTrack 2017's 15 joints -> the 17-slot ordering used by this repository.
# Ear slots do not exist in PoseTrack and remain zero.
PT_TO_MODEL = {
    13: 0, 12: 1, 14: 2,
    9: 5, 8: 6, 10: 7, 7: 8, 11: 9, 6: 10,
    3: 11, 2: 12, 4: 13, 1: 14, 5: 15, 0: 16,
}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return [item for element in value.ravel().tolist() for item in as_list(element)]
    return [value]


def native(value):
    """Make scipy/numpy annotation values JSON serializable."""
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.ravel().tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def person_keypoints(person):
    keypoints = [0.0] * (17 * 3)
    if not isinstance(person, dict):
        return keypoints
    annopoints = person.get("annopoints")
    if not isinstance(annopoints, dict):
        return keypoints
    points = as_list(annopoints.get("point"))
    for point in points:
        if not isinstance(point, dict):
            continue
        target = PT_TO_MODEL.get(int(point["id"]))
        if target is None:
            continue
        visible = point.get("is_visible", 1)
        visible = int(visible) if visible is not None else 1
        base = target * 3
        keypoints[base:base + 3] = [float(point["x"]), float(point["y"]), 2 if visible else 1]
    return keypoints


def pose_bbox(keypoints, width, height):
    xs = [keypoints[i] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    ys = [keypoints[i + 1] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    if not xs:
        return None
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    # Match the common PoseTrack conversion convention: pad the tight pose box.
    padx = max(10.0, (x2 - x1) * 0.15)
    pady = max(10.0, (y2 - y1) * 0.15)
    x1, y1 = max(0.0, x1 - padx), max(0.0, y1 - pady)
    x2, y2 = min(float(width - 1), x2 + padx), min(float(height - 1), y2 + pady)
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def category():
    return {
        "supercategory": "person", "id": 1, "name": "person",
        "keypoints": KEYPOINT_NAMES,
        "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                     [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                     [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                     [4, 6], [5, 7]],
    }


def convert_split(dataset_root, split, output_root):
    mat_dir = dataset_root / "annotations" / split
    image_root = dataset_root / "images"
    renamed_image_root = output_root / "images_renamed"
    images, annotations = [], []
    image_id = annotation_id = 1
    poseval_dir = output_root / "poseval" / split
    poseval_dir.mkdir(parents=True, exist_ok=True)

    for mat_path in sorted(mat_dir.glob("*.mat")):
        frames = as_list(loadmat(mat_path, simplify_cells=True)["annolist"])
        poseval_frames = []
        for frame in frames:
            original_name = frame["image"]["name"]
            relative_name = original_name.removeprefix("images/")
            image_path = image_root / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as im:
                width, height = im.size
            frame_id = int(frame.get("imgnum", Path(relative_name).stem))
            renamed_name = Path(relative_name).parent / f"{frame_id:08d}.jpg"
            renamed_path = renamed_image_root / renamed_name
            renamed_path.parent.mkdir(parents=True, exist_ok=True)
            if renamed_path.is_symlink() or renamed_path.exists():
                if renamed_path.resolve() != image_path.resolve():
                    raise RuntimeError(f"Conflicting renamed frame: {renamed_path}")
            else:
                renamed_path.symlink_to(image_path)
            current_image_id = image_id
            images.append({
                "id": current_image_id, "file_name": str(renamed_name),
                "width": width, "height": height, "nframes": len(frames),
                "frame_id": frame_id, "is_labeled": int(frame.get("is_labeled", 0)),
            })
            image_id += 1

            for person in as_list(frame.get("annorect")):
                if not isinstance(person, dict):
                    continue
                keypoints = person_keypoints(person)
                bbox = pose_bbox(keypoints, width, height)
                if bbox is None:
                    continue
                annotations.append({
                    "id": annotation_id, "image_id": current_image_id,
                    "category_id": 1, "iscrowd": 0, "bbox": bbox,
                    "area": bbox[2] * bbox[3], "keypoints": keypoints,
                    "num_keypoints": sum(keypoints[i] > 0 for i in range(2, len(keypoints), 3)),
                    "track_id": int(person.get("track_id", -1)),
                })
                annotation_id += 1

            poseval_frames.append({
                "image": {"name": f"images/{renamed_name}"},
                "imgnum": [frame_id],
                "is_labeled": [int(frame.get("is_labeled", 0))],
                "annorect": native(as_list(frame.get("annorect"))),
            })

        (poseval_dir / f"{mat_path.stem}.json").write_text(
            json.dumps({"annolist": poseval_frames}, separators=(",", ":")),
            encoding="utf-8",
        )

    result = {"images": images, "annotations": annotations, "categories": [category()]}
    json_dir = output_root / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)
    output_file = json_dir / f"posetrack_{split}.json"
    output_file.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"{split}: {len(images)} images, {len(annotations)} poses -> {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Directory containing images/ and annotations/")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for split in ("train", "val"):
        convert_split(args.dataset_root, split, args.output_root)


if __name__ == "__main__":
    main()
