import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import copy
import random
from collections import defaultdict, Counter
from torch.utils.data import Subset
from typing import Tuple, Optional, Dict
import pandas as pd
import math
from scipy.optimize import linear_sum_assignment
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
from typing import Tuple, Optional, Dict

# =============================================================================
# Constants & Config
# =============================================================================

DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_ROI = {"tl": (465, 126), "tr": (1300, 139), "br": (1256, 944), "bl": (483, 930)}

INSTANCE_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
]

# =============================================================================
# Group 1: Reading & Data Structures
# =============================================================================

class YOLOPoseData:
    def __init__(self, data_line):
        # cls
        self.cls = int(data_line[0]) if data_line[0] is not None else None

        # bbox (xywh)
        self.xywh = data_line[1:5]
        self.xy = data_line[1:3]

        if len(data_line)%3 == 0:
            # id
            self.id = int(data_line[-1]) if data_line[-1] is not None else None

            # Parse keypoints
            points_line = data_line[5:-1]
        else:
            self.id = None

            # Parse keypoints
            points_line = data_line[5:]

        self.points_num = len(points_line) // 3
        self.points_xy = []
        self.points_score = []

        # Initialize kinematics and angle related properties (keep as-is)
        self.xy_dot = []
        self.xy_ddot = []
        self.points_xy_dot = []
        self.points_xy_ddot = []
        self.angles = []
        self.angles_dot = []
        self.angles_ddot = []

        # Iterate through each keypoint
        for i in range(self.points_num):
            x_raw = points_line[i * 3]
            y_raw = points_line[i * 3 + 1]
            vis_flag = points_line[i * 3 + 2]

            # Case 1: [0, 0, 0] → completely missing
            if x_raw == 0 and y_raw == 0 and vis_flag == 0:
                self.points_xy.append([None, None])
                self.points_score.append(None)
            # Case 2: vis_flag == 2 → visible point (score=1)
            elif vis_flag == 2:
                self.points_xy.append([float(x_raw), float(y_raw)])
                self.points_score.append(1.0)
            # Case 3: other (e.g. floating point score from predictions, or outliers) → keep as-is
            else:
                # Assume this is a model prediction (e.g. [x, y, 0.87]), keep as-is
                self.points_xy.append([float(x_raw), float(y_raw)] if x_raw is not None else [None, None])
                self.points_score.append(float(vis_flag) if vis_flag is not None else None)

    def normalize_point(self, ref_point, point):
        """
        Convert point to offset coordinates relative to ref_point.

        Args:
            ref_point (list or tuple): Reference point [x0, y0]
            point (list or tuple): Input point [x, y]

        Returns:
            list: Relative coordinates [x - x0, y - y0]
        """
        x0, y0 = ref_point
        x, y = point
        return [x - x0, y - y0]

def load_pose_data(folder_path):
    """
    Read all YOLO Pose format txt files in the specified folder,
    return a list: [frame][instance][17 values]
    Frames sorted by frame number (starting from 1).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Path {folder_path} is not a valid folder")

    # Get all .txt files
    txt_files = list(folder.glob("*.txt"))
    if not txt_files:
        print("Warning: No .txt files found in the folder")
        return []

    # Store per-frame data using a dict keyed by frame number
    frame_dict = {}

    # Wrap file list with tqdm to show progress bar
    for txt_file in tqdm(txt_files, desc="Loading Pose data", unit="file"):
        name = txt_file.stem
        match = re.match(r"(.+?)_(\d+)$", name)
        if not match:
            # Optionally: log skipped files, but do not interrupt
            continue

        frame_num = int(match.group(2))
        instances = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                values = list(map(float, line.split()))
                instances.append(values)
        frame_dict[frame_num] = instances  # ← All rows included!

    if not frame_dict:
        return []

    max_frame = max(frame_dict.keys())
    min_frame = min(frame_dict.keys())

    # Note: Assumes frames start from 1 and are continuous, but in practice they may not be
    # If you want strict 1,2,3,... sequence with missing frames filled by empty lists:
    result = []
    for i in range(min_frame, max_frame + 1):
        result.append(frame_dict.get(i, []))  # Missing frames filled with empty list

    return result

def load_pose_data_by_file_order(folder_path):
    """
    Read all YOLO Pose format txt files in the specified folder,
    sorted by filename (lexicographic order), each file treated as one frame,
    returns: List[List[instance]], i.e. [frame_idx][instance][values]
    Frame indices start from 0 and are continuous.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Path {folder_path} is not a valid folder")

    # Get all .txt files, sorted by filename (lexicographic order)
    txt_files = sorted(folder.glob("*.txt"), key=lambda x: x.name)

    if not txt_files:
        print("Warning: No .txt files found in the folder")
        return []

    result = []
    for txt_file in tqdm(txt_files, desc="Loading Pose data (by file order)", unit="file"):
        instances = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = list(map(float, line.split()))
                    instances.append(values)
        result.append(instances)  # Each file = one frame

    return result

def process_pose_file(file_path, max_instance_num, is_mouse=True):
    """
    Process a single pose file (mouse or tail)
    Returns:
        List[List[YOLOPoseData | None]]  # sorted by frame and slot
    """
    org_data = load_pose_data(file_path)
    sorted_data = []
    last_frame_ids = [None] * max_instance_num

    for frame in org_data:
        current_instances = [YOLOPoseData(inst) for inst in frame]
        new_frame = [None] * max_instance_num
        match = [False] * max_instance_num

        # Step 1: Match existing IDs
        for inst in current_instances:
            matched = False
            for slot_idx, last_id in enumerate(last_frame_ids):
                if last_id is not None and inst.id == last_id:
                    new_frame[slot_idx] = inst
                    match[slot_idx] = True
                    matched = True
                    break
            if matched:
                continue
            # Step 2: Place into free slot
            for slot_idx in range(max_instance_num):
                if not match[slot_idx]:
                    new_frame[slot_idx] = inst
                    match[slot_idx] = True
                    break

        current_frame_ids = [inst.id if inst else None for inst in new_frame]
        last_frame_ids = current_frame_ids
        sorted_data.append(new_frame)

    return sorted_data

def read_keypoints_by_file(
        features,
        seq_length=5,
        stride=5,
        batch_size=16,
        device='cuda',
        pad_last_batch=True
):
    """
    Slice sliding windows from a feature tensor and pack them into batches of specified batch_size.

    Args:
        features (torch.Tensor): [T, D]
        seq_length (int): Window length
        stride (int): Sliding stride
        batch_size (int): Batch size (e.g. 16)
        device (str): Output device
        pad_last_batch (bool): Whether to zero-pad the final incomplete batch

    Returns:
        windows_tensor (torch.Tensor): [num_batches, batch_size, seq_length, D] on device
        center_indices_batches (List[List[int]]): Center frame indices per batch, -1 for padded positions
        total_frames (int)
    """
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")

    T, D = features.shape
    if T < seq_length:
        raise ValueError(f"Sequence too short ({T} < {seq_length})")

    # Step 1: Extract all windows and center frame indices (CPU numpy)
    features_np = features.detach().cpu().numpy()
    all_windows = []  # list of [seq_len, D]
    all_centers = []  # list of int

    start = 0
    while start + seq_length <= T:
        window = features_np[start:start + seq_length]  # [seq_len, D]
        all_windows.append(window)
        all_centers.append(start + seq_length // 2)
        start += stride

    N = len(all_windows)
    if N == 0:
        # Construct empty output
        windows_tensor = torch.empty((0, batch_size, seq_length, D), dtype=features.dtype, device=device)
        center_indices_batches = []
        return windows_tensor, center_indices_batches, T

    # Step 2: Group by batch_size
    num_full_batches = N // batch_size
    remainder = N % batch_size

    windows_batches = []
    center_batches = []

    # Process full batches
    for i in range(num_full_batches):
        batch_wins = np.stack(all_windows[i * batch_size: (i + 1) * batch_size], axis=0)  # [B, seq_len, D]
        batch_centers = all_centers[i * batch_size: (i + 1) * batch_size]  # [B]
        windows_batches.append(batch_wins)
        center_batches.append(batch_centers)

    # Process the final incomplete batch
    if remainder > 0:
        if pad_last_batch:
            # Take remaining windows
            last_windows = all_windows[num_full_batches * batch_size:]  # [R, seq_len, D], R < B
            last_centers = all_centers[num_full_batches * batch_size:]  # [R]

            # pad to batch_size
            pad_count = batch_size - remainder
            pad_window = np.zeros((pad_count, seq_length, D), dtype=features_np.dtype)
            pad_centers = [-1] * pad_count

            padded_windows = np.concatenate([last_windows, pad_window], axis=0)  # [B, seq_len, D]
            padded_centers = last_centers + pad_centers  # [B]

            windows_batches.append(padded_windows)
            center_batches.append(padded_centers)
        else:
            # Discard the final incomplete batch
            pass

    # Step 3: Merge all batches
    if windows_batches:
        all_batches_np = np.stack(windows_batches, axis=0)  # [num_batches, B, seq_len, D]
        windows_tensor = torch.from_numpy(all_batches_np).to(device)
    else:
        windows_tensor = torch.empty((0, batch_size, seq_length, D), dtype=features.dtype, device=device)
        center_batches = []

    return windows_tensor, center_batches, T

def save_dataset_to_ts(
        dataset,
        output_path: str,
        num_channels: int,
        class_names=None,
        layout="time_last",  # or "time_first"
        has_timestamps=False,
        allow_missing=False,
        problem_name="CustomDataset"
):
    """
    Save any torch.utils.data.Dataset instance to UEA/UCR format .ts file.

    Args:
        dataset: Instance inheriting from torch.utils.data.Dataset, __getitem__ returns (x, y)
        output_path: Output file path, e.g. "mydata_TRAIN.ts"
        num_channels: Number of time series channels (feature dimension)
        class_names: Optional list of class names, e.g. ["cat", "dog"]; if not provided, inferred from integer labels
        layout:
            - "time_last": x.shape = (seq_len, num_channels)  ← common in PyTorch
            - "time_first": x.shape = (num_channels, seq_len)
        has_timestamps: Whether timestamps are included (almost always false in .ts format)
        allow_missing: Whether NaN is allowed (if True, NaN is written as "NaN")
        problem_name: Dataset name (written to @problemName)
    """
    n_samples = len(dataset)

    # Collect all samples to infer classes and sequence length (optional optimization: read once)
    all_labels = []
    max_seq_len = 0
    for i in range(min(10, n_samples)):  # Quick probe


        x, y = dataset[i]
        x.cpu()
        x = np.array(x)
        if layout == "time_first":
            x = x.T  # Convert to (seq_len, num_channels)
        max_seq_len = max(max_seq_len, x.shape[0])
        all_labels.append(int(y))

    all_labels = class_names
    unique_labels = sorted(set(all_labels))
    if class_names is None:
        class_names = [str(l) for l in unique_labels]
    else:
        assert len(class_names) >= len(unique_labels), "class_names too short"


    # Start writing file
    with open(output_path, 'w') as f:
        f.write(f"@problemName {problem_name}\n")
        f.write(f"@timeStamps {str(has_timestamps).lower()}\n")
        f.write(f"@univariate {str(num_channels == 1).lower()}\n")
        f.write(f"@dimension {num_channels}\n")
        f.write(f"@missing {str(allow_missing).lower()}\n")
        f.write(f"@classlabel true {' '.join(str(l) for l in unique_labels)}\n")
        f.write("@data\n")

        for i in range(n_samples):
            x, y = dataset[i]
            x = np.array(x)

            # Unify to (seq_len, num_channels)
            if layout == "time_first":
                x = x.T

            seq_len, d = x.shape
            assert d == num_channels, f"Sample {i}: expected {num_channels} channels, got {d}"

            # Handle missing values
            if allow_missing:
                time_steps = []
                for t in range(seq_len):
                    feats = []
                    for c in range(num_channels):
                        val = x[t, c]
                        if np.isnan(val):
                            feats.append("NaN")
                        else:
                            feats.append(f"{val:.6f}")
                    time_steps.append(','.join(feats))
            else:
                if np.isnan(x).any():
                    raise ValueError(f"Sample {i} contains NaN, but allow_missing=False")
                time_steps = [','.join([f"{val:.6f}" for val in x[t]]) for t in range(seq_len)]

            line = ' : '.join(time_steps) + f' : {int(y)}\n'
            f.write(line)

    print(f"Saved {n_samples} samples to {output_path}")

def create_balanced_dataset(
    dataset: 'MouseBehaviorDataset',
    max_ratio: float = 2.0,
    random_seed: Optional[int] = 42,
    shuffle: bool = False,
    verbose: bool = True
) -> Subset:
    """
    Extract all samples from MouseBehaviorDataset,
    perform constrained downsampling per class (keep at most max_ratio * min_class_count samples),
    optionally shuffle, and return a class-balanced Subset.

    If shuffle=False, preserve temporal order within each class.
    """
    if random_seed is not None:
        import random
        random.seed(random_seed)

    # Step 1: Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)

    # Step 2: Group indices by class (preserve original order)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        label_to_indices[label].append(idx)

    # Step 3: Compute minimum class count & maximum allowed count
    class_counts = {label: len(indices) for label, indices in label_to_indices.items()}
    min_count = min(class_counts.values())
    max_allowed = int(max_ratio * min_count)

    if verbose:
        print(f"[Original class counts]: {class_counts}")
        print(f"Min class count: {min_count}, Max allowed per class: {max_allowed} (ratio={max_ratio})")

    # Step 4: Constrained sampling per class
    selected_indices = []
    final_class_counts = {}

    for label, indices in label_to_indices.items():
        current_count = len(indices)
        target_count = min(current_count, max_allowed)
        if shuffle:
            import random
            sampled = random.sample(indices, target_count)
        else:
            sampled = indices[:target_count]  # Preserve temporal order
        selected_indices.extend(sampled)
        final_class_counts[label] = target_count

    # Step 5: Optionally shuffle overall order
    if shuffle:
        import random
        random.shuffle(selected_indices)

    balanced_dataset = Subset(dataset, selected_indices)

    if verbose:
        labels = [balanced_dataset[i][1] for i in range(len(balanced_dataset))]
        dist = dict(Counter(labels))
        print(f"[Balanced dataset class counts]: {dist}")
        print(f"Total selected samples: {len(balanced_dataset)}")

    return balanced_dataset

def compute_class_distribution_from_loader(loader: DataLoader) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Compute the sample count and ratio for each class in the DataLoader.

    Args:
        loader (DataLoader): Input DataLoader, whose dataset __getitem__ should return (data, label)

    Returns:
        Tuple[Dict[int, int], Dict[int, float]]:
            - First dict: {class: sample count}
            - Second dict: {class: ratio (0~1)}
    """
    all_labels = []

    # Iterate through the entire DataLoader to get all labels
    for _, labels in loader:
        # labels may be tensor or list, unify to Python int list
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().tolist()  # If tensor
        else:
            labels = list(labels)  # If list or other iterable type
        all_labels.extend(labels)

    total = len(all_labels)
    if total == 0:
        return {}, {}

    count_dict = dict(Counter(all_labels))
    ratio_dict = {cls: count / total for cls, count in count_dict.items()}

    return count_dict, ratio_dict

# =============================================================================
# Group 2: Visualization
# =============================================================================

def draw_star(image, center, radius, color, thickness=-1):
    """
    Draw a filled five-pointed star (using polygon approximation)
    """
    cx, cy = center
    points = []
    for i in range(5):
        angle = np.deg2rad(90 + i * 72)
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        points.append([x, y])
        inner_angle = np.deg2rad(90 + i * 72 + 36)
        x_inner = int(cx + radius * 0.382 * np.cos(inner_angle))
        y_inner = int(cy + radius * 0.382 * np.sin(inner_angle))
        points.append([x_inner, y_inner])
    pts = np.array([points], dtype=np.int32)
    if thickness == -1:
        cv2.fillPoly(image, pts, color)
    else:
        cv2.polylines(image, pts, isClosed=True, color=color, thickness=thickness)

def visualize_frame(frame_img, merged_instances, frame_idx, max_instance_num=2):
    img = frame_img.copy()
    h, w = img.shape[:2]

    # Draw frame index
    cv2.putText(img, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    point_positions = {}  # Track number of instances at each position

    for inst_idx, inst in enumerate(merged_instances):
        if inst is None:
            continue

        color = INSTANCE_COLORS[inst_idx % len(INSTANCE_COLORS)]

        # Draw detection box
        x_norm, y_norm, w_norm, h_norm = inst.xywh
        cx = x_norm * w
        cy = y_norm * h
        box_w = w_norm * w
        box_h = h_norm * h
        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        default_radius = 5
        star_radius = int(default_radius * 1.6)  # Star slightly larger, more visible

        for pt_idx, (px_norm, py_norm) in enumerate(inst.points_xy):
            if px_norm is None or py_norm is None:
                continue

            px = px_norm * w
            py = py_norm * h
            px_int, py_int = int(px), int(py)
            if not (0 <= px_int < w and 0 <= py_int < h):
                continue

            pos_key = f"{px_int}_{py_int}"
            if pos_key not in point_positions:
                point_positions[pos_key] = []

            point_positions[pos_key].append((inst_idx, pt_idx, color))

    # Draw keypoints
    for pos_key, instances in point_positions.items():
        px_int, py_int = map(int, pos_key.split('_'))
        if len(instances) > 1:  # If overlapping
            # Draw enlarged stars in instance order
            for idx, (inst_idx, pt_idx, color) in enumerate(instances):
                adjusted_radius = star_radius + idx * 2  # Scale factor adjustable
                draw_star(img, (px_int, py_int), adjusted_radius, color)
        else:
            inst_idx, pt_idx, color = instances[0]
            if pt_idx in [3, 4]:  # 4th and 5th keypoints (0-indexed)
                draw_star(img, (px_int, py_int), star_radius, color)
            else:
                cv2.circle(img, (px_int, py_int), default_radius, color, -1)

    return img

def visualize_merged_data(
        Merged_data,
        video_paths,
        output_dir="./output_vis",
        max_instance_num=2,
        codec='XVID',
        output_mode='video'  # 'video' or 'images'
):
    """
    Visualize merged data, supporting output as video or image sequence.

    Args:
        Merged_data: List[List[List[YOLOPoseData | None]]]
        video_paths: List[str]
        output_dir: Output root directory
        max_instance_num: Maximum instances per frame
        codec: Video codec (only used in 'video' mode)
        output_mode: 'video' or 'images'
    """
    assert output_mode in ['video', 'images'], "output_mode must be 'video' or 'images'"
    os.makedirs(output_dir, exist_ok=True)

    for file_idx, (merged_file, video_path) in enumerate(zip(Merged_data, video_paths)):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Unable to open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_data = len(merged_file)

        base_name = os.path.splitext(os.path.basename(video_path))[0]  # Remove extension

        if output_mode == 'video':
            # Construct output video path (preserve original extension)
            ext = os.path.splitext(video_path)[1].lower()
            if ext in ['.avi', '.mp4']:
                output_path = os.path.join(output_dir, f"vis_{os.path.basename(video_path)}")
            else:
                output_path = os.path.join(output_dir, f"vis_{base_name}.avi")

            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"\nGenerating video [{file_idx + 1}/{len(video_paths)}]: {os.path.basename(output_path)}")

        elif output_mode == 'images':
            # Create separate subfolder for each video
            img_output_dir = os.path.join(output_dir, f"vis_{base_name}_frames")
            os.makedirs(img_output_dir, exist_ok=True)
            print(f"\nSaving image sequence [{file_idx + 1}/{len(video_paths)}]: {img_output_dir}")

        frame_iter = min(total_frames_video, total_frames_data)
        with tqdm(total=frame_iter, desc=f"  {base_name}", unit="frame") as pbar:
            for frame_idx in range(frame_iter):
                ret, frame = cap.read()
                if not ret:
                    break

                merged_frame = merged_file[frame_idx]
                if len(merged_frame) < max_instance_num:
                    merged_frame = merged_frame + [None] * (max_instance_num - len(merged_frame))
                else:
                    merged_frame = merged_frame[:max_instance_num]

                vis_frame = visualize_frame(frame, merged_frame, frame_idx, max_instance_num)

                if output_mode == 'video':
                    out.write(vis_frame)
                elif output_mode == 'images':
                    img_path = os.path.join(img_output_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(img_path, vis_frame)

                pbar.update(1)

        cap.release()
        if output_mode == 'video':
            out.release()
            print(f"Video saved: {output_path}")
        elif output_mode == 'images':
            print(f"Saved {frame_iter} images to: {img_output_dir}")

# =============================================================================
# Group 3: Helper Calculation (Vectorized)
# =============================================================================

def set_default_roi(image_width: int= None, image_height: int= None, roi: dict= None):
    global DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_ROI
    if image_width is not None:
        DEFAULT_IMAGE_WIDTH = image_width
    if image_height is not None:
        DEFAULT_IMAGE_HEIGHT = image_height
    if roi is not None:
        DEFAULT_ROI = roi

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points [x, y], supports None values (returns inf)"""
    if p1 is None or p2 is None or p1[0] is None or p1[1] is None or p2[0] is None or p2[1] is None:
        return float('inf')
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def sanitize_points(points, expected_len=7):
    """
    Convert points to a list of length expected_len.
    - Valid points: [x, y] (float)
    - Invalid/missing points: None (not [0,0]!)
    """
    if points is None:
        return [None] * expected_len

    result = []
    for i in range(expected_len):
        if i < len(points) and points[i] is not None:
            pt = points[i]
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                x, y = pt[0], pt[1]
                if x is not None and y is not None:
                    try:
                        result.append([float(x), float(y)])
                    except (TypeError, ValueError):
                        result.append(None)
                else:
                    result.append(None)
            else:
                result.append(None)
        else:
            result.append(None)
    return result

def video_to_tensor(video_data, max_instance_num, num_keypoints, device='cpu'):
    """
    Convert a single video's data to PyTorch tensors.
    Returns:
        points_tensor: [T, M, K, 2]
        box_tensor:    [T, M, 4]  (xywh)
        mask_tensor:   [T, M]     (bool, True if instance exists)
    """
    T = len(video_data)
    M = max_instance_num
    K = num_keypoints

    points_np = np.zeros((T, M, K, 2), dtype=np.float32)
    box_np = np.zeros((T, M, 4), dtype=np.float32)
    mask_np = np.zeros((T, M), dtype=bool)

    for t, frame in enumerate(video_data):
        for m in range(M):
            inst = frame[m] if m < len(frame) else None
            if inst:
                mask_np[t, m] = True
                # Box
                if inst.xywh and len(inst.xywh) >= 4:
                    box_np[t, m] = [float(x) if x is not None else 0.0 for x in inst.xywh]

                # Points
                if inst.points_xy:
                    for k in range(min(K, len(inst.points_xy))):
                        pt = inst.points_xy[k]
                        if pt and len(pt) >= 2 and pt[0] is not None and pt[1] is not None:
                            points_np[t, m, k, 0] = float(pt[0])
                            points_np[t, m, k, 1] = float(pt[1])

    return (torch.from_numpy(points_np).to(device),
            torch.from_numpy(box_np).to(device),
            torch.from_numpy(mask_np).to(device))

def compute_kinematics_vectorized(pos_tensor, fps, window_size=5):
    """
    Compute velocity and acceleration.
    pos_tensor: [T, ..., D]
    Returns: vel [T, ..., D], acc [T, ..., D]
    Uses 5-point central difference.
    """
    dt = 1.0 / fps
    dt2 = dt * dt

    if pos_tensor.shape[0] < window_size:
        # Fallback to simple diff
        vel = torch.zeros_like(pos_tensor)
        vel[1:] = (pos_tensor[1:] - pos_tensor[:-1]) / dt
        acc = torch.zeros_like(pos_tensor)
        acc[2:] = (pos_tensor[2:] - 2*pos_tensor[1:-1] + pos_tensor[:-2]) / dt2
        return vel, acc

    # 5-point stencil for 1st derivative (Velocity)
    # v[t] = (-x[t+2] + 8x[t+1] - 8x[t-1] + x[t-2]) / (12*dt)
    vel = torch.zeros_like(pos_tensor)
    vel[2:-2] = (-pos_tensor[4:] + 8*pos_tensor[3:-1] - 8*pos_tensor[1:-3] + pos_tensor[:-4]) / (12 * dt)

    # Boundaries (Simple diff)
    vel[1] = (pos_tensor[2] - pos_tensor[0]) / (2 * dt)
    vel[-2] = (pos_tensor[-1] - pos_tensor[-3]) / (2 * dt)
    vel[0] = (pos_tensor[1] - pos_tensor[0]) / dt
    vel[-1] = (pos_tensor[-1] - pos_tensor[-2]) / dt

    # Acceleration: derivative of velocity
    acc = torch.zeros_like(vel)
    acc[2:-2] = (-vel[4:] + 8*vel[3:-1] - 8*vel[1:-3] + vel[:-4]) / (12 * dt)

    # Boundaries
    acc[1] = (vel[2] - vel[0]) / (2 * dt)
    acc[-2] = (vel[-1] - vel[-3]) / (2 * dt)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[-2]) / dt

    return vel, acc

def compute_angles_vectorized(points_tensor):
    """
    Compute angles between 3 consecutive points.
    points_tensor: [T, M, K, 2]
    Returns: angles [T, M, K-2]
    """
    T, M, K, _ = points_tensor.shape
    if K < 3:
        return torch.zeros((T, M, 0), device=points_tensor.device)

    # Vectors: v1 = p[k] - p[k+1], v2 = p[k+2] - p[k+1]
    # p_a=k, p_b=k+1, p_c=k+2. Angle at p_b.

    p_a = points_tensor[:, :, 0:K-2, :] # [T, M, K-2, 2]
    p_b = points_tensor[:, :, 1:K-1, :]
    p_c = points_tensor[:, :, 2:K, :]

    v1 = p_a - p_b
    v2 = p_c - p_b

    # Norms
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)

    # Dot product
    dot = (v1 * v2).sum(dim=-1)

    # Cosine
    denom = norm1 * norm2 + 1e-8
    cos_angle = torch.clamp(dot / denom, -1.0, 1.0)

    angles = torch.acos(cos_angle) # Radians

    return angles

def avg_pairwise_dist_vectorized(pts):
    # pts: [T, K, 2]
    # return: [T]
    T, K, _ = pts.shape
    if K < 2:
        return torch.zeros(T, device=pts.device)

    # Expand dims for broadcasting: [T, K, 1, 2] - [T, 1, K, 2]
    diff = pts.unsqueeze(2) - pts.unsqueeze(1) # [T, K, K, 2]
    dist = torch.norm(diff, dim=-1) # [T, K, K]

    # Upper triangular indices
    triu_indices = torch.triu_indices(K, K, offset=1)
    pairwise_dists = dist[:, triu_indices[0], triu_indices[1]] # [T, K*(K-1)/2]

    return pairwise_dists.mean(dim=1)

# =============================================================================
# Group 4: Vector Assembly & Main Classes
# =============================================================================

class VectorizedContext:
    """
    Holds tensors for a single video to avoid re-computation.
    """
    def __init__(self, points_t, box_t, mask_t, fps, num_keypoints, device):
        self.points_t = points_t # [T, M, K, 2]
        self.box_t = box_t       # [T, M, 4]
        self.mask_t = mask_t     # [T, M]
        self.fps = fps
        self.num_keypoints = num_keypoints
        self.device = device
        self.T, self.M, self.K, _ = points_t.shape

        # Lazy cache
        self._points_vel = None
        self._points_acc = None
        self._box_vel = None
        self._box_acc = None
        self._angles = None

    @property
    def points_vel(self):
        if self._points_vel is None:
            self._points_vel, self._points_acc = compute_kinematics_vectorized(self.points_t, self.fps, window_size=5)

            # Also filter keypoint velocities
            vel_norm = torch.norm(self._points_vel, dim=-1, keepdim=True)
            mask = vel_norm < 0.05
            self._points_vel = torch.where(mask, torch.zeros_like(self._points_vel), self._points_vel)

        return self._points_vel

    @property
    def points_acc(self):
        if self._points_acc is None:
            _ = self.points_vel
        return self._points_acc

    @property
    def box_vel(self):
        if self._box_vel is None:
            self._box_vel, self._box_acc = compute_kinematics_vectorized(self.box_t[..., :2], self.fps, window_size=5)

            # Filter tiny oscillations: if velocity magnitude is below threshold, set to zero
            # Note: box_t is normalized coordinates (0-1), 0.05 means 5% of image width per second
            # If fps=30, movement per frame is 0.05/30 ~= 0.0016, indeed very small
            vel_norm = torch.norm(self._box_vel, dim=-1, keepdim=True)
            mask = vel_norm < 0.05
            self._box_vel = torch.where(mask, torch.zeros_like(self._box_vel), self._box_vel)

        return self._box_vel

    @property
    def box_acc(self):
        if self._box_acc is None:
            _ = self.box_vel
        return self._box_acc

    @property
    def angles(self):
        if self._angles is None:
            self._angles = compute_angles_vectorized(self.points_t)
        return self._angles

# --- Feature Calculators (v1.3: 42D skeleton + motion + tail + social) ---
# Keypoint mapping with 10 points (7 mouse + 3 tail):
#   0=snout, 1=head_center, 2=body_center, 3=tailbase,
#   4=left_ear, 5=right_ear, 6=unused,
#   7=tail_mid1, 8=tail_mid2, 9=tail_tip

import math as _math

def _skeleton_features(pts, center_id):
    """6D: nose-head, head-body, body-tail, orientation, length, compactness"""
    nose   = pts[:, center_id, 0, :]   # [T, 2]
    head   = pts[:, center_id, 1, :]
    body   = pts[:, center_id, 2, :]
    tailb  = pts[:, center_id, 3, :]

    n_to_h = torch.norm(nose - head, dim=1, keepdim=True)
    h_to_b = torch.norm(head - body, dim=1, keepdim=True)
    b_to_t = torch.norm(body - tailb, dim=1, keepdim=True)
    length  = torch.norm(nose - tailb, dim=1, keepdim=True)

    body_vec = head - body
    orient = torch.atan2(body_vec[:, 1], body_vec[:, 0]).unsqueeze(1)

    # compactness: avg pairwise distance of nose,head,body,tailbase
    kpts = pts[:, center_id, :4, :]  # [T, 4, 2]
    diff = kpts.unsqueeze(2) - kpts.unsqueeze(1)
    dist_mat = torch.norm(diff, dim=-1)
    triu_mask = torch.triu(torch.ones(4, 4, device=kpts.device), diagonal=1)
    compact = (dist_mat * triu_mask.unsqueeze(0)).sum(dim=(1,2)) / triu_mask.sum()

    return torch.cat([n_to_h, h_to_b, b_to_t, orient, length, compact.unsqueeze(1)], dim=1)

def _motion_features(ctx, cid):
    """8D per mouse: head_vel, body_vel, tail_vel, speed, accel"""
    h_vel = ctx.points_vel[:, cid, 1, :]   # kp1=head
    b_vel = ctx.points_vel[:, cid, 2, :]   # kp2=body
    t_vel = ctx.points_vel[:, cid, 3, :]   # kp3=tailbase
    speed = torch.norm(ctx.box_vel[:, cid, :2], dim=1, keepdim=True)
    accel = torch.norm(ctx.box_acc[:, cid, :2], dim=1, keepdim=True)
    return torch.cat([h_vel, b_vel, t_vel, speed, accel], dim=1)

def _tail_features(ctx, cid):
    """3D per mouse: tail_angle, tail_curve, tail_motion"""
    pts = ctx.points_t
    tailb = pts[:, cid, 3, :]; head = pts[:, cid, 1, :]; body = pts[:, cid, 2, :]
    t_mid = pts[:, cid, 7, :]; t_tip = pts[:, cid, 9, :]
    body_vec = head - body
    body_ang = torch.atan2(body_vec[:,1], body_vec[:,0])
    tail_vec = t_tip - tailb
    tail_ang = torch.atan2(tail_vec[:,1], tail_vec[:,0])
    angle = ((tail_ang - body_ang + _math.pi) % (2*_math.pi) - _math.pi).unsqueeze(1)
    mid_exp = (tailb + t_tip) / 2
    curve = torch.norm(t_mid - mid_exp, dim=1, keepdim=True)
    motion = torch.norm(ctx.points_vel[:, cid, 9, :], dim=1, keepdim=True)
    return torch.cat([angle, curve, motion], dim=1)

def feat_skeleton(ctx, center_id):
    other_id = 1 - center_id
    s = _skeleton_features(ctx.points_t, center_id)
    o = _skeleton_features(ctx.points_t, other_id)
    return torch.cat([s, o], dim=1), [
        'nose_to_head','head_to_body','body_to_tail','body_orientation','body_length','body_compactness',
        'other_nose_to_head','other_head_to_body','other_body_to_tail',
        'other_body_orientation','other_body_length','other_body_compactness']

def feat_motion(ctx, center_id):
    other_id = 1 - center_id
    s = _motion_features(ctx, center_id)
    o = _motion_features(ctx, other_id)
    return torch.cat([s, o], dim=1), [
        'head_vel_x','head_vel_y','body_vel_x','body_vel_y','tail_vel_x','tail_vel_y','speed','acceleration',
        'other_head_vel_x','other_head_vel_y','other_body_vel_x','other_body_vel_y',
        'other_tail_vel_x','other_tail_vel_y','other_speed','other_acceleration']

def feat_tail(ctx, center_id):
    other_id = 1 - center_id
    s = _tail_features(ctx, center_id)
    o = _tail_features(ctx, other_id)
    return torch.cat([s, o], dim=1), [
        'tail_angle','tail_curve','tail_motion',
        'other_tail_angle','other_tail_curve','other_tail_motion']

def feat_social(ctx, center_id):
    """8D: dist, self_facing, other_facing, mutual, approach, rel_speed, heading_diff, axis_align"""
    other_id = 1 - center_id
    pts = ctx.points_t
    diff = ctx.box_t[:, other_id, :2] - ctx.box_t[:, center_id, :2]
    dist = torch.norm(diff, dim=1, keepdim=True)

    s_head = pts[:, center_id, 1, :]; s_body = pts[:, center_id, 2, :]
    o_head = pts[:, other_id, 1, :];   o_body = pts[:, other_id, 2, :]
    s_dir = s_head - s_body; o_dir = o_head - o_body
    s_ang = torch.atan2(s_dir[:,1], s_dir[:,0])
    o_ang = torch.atan2(o_dir[:,1], o_dir[:,0])

    to_other = pts[:, other_id, 2, :] - pts[:, center_id, 2, :]
    to_o_ang = torch.atan2(to_other[:,1], to_other[:,0])
    self_facing = torch.cos(s_ang - to_o_ang).unsqueeze(1)
    other_facing = torch.cos(o_ang - (to_o_ang + _math.pi)).unsqueeze(1)
    mutual = self_facing * other_facing

    sv = ctx.box_vel[:, center_id, :2]; ov = ctx.box_vel[:, other_id, :2]
    rel_v = sv - ov
    approach = -(rel_v * diff / (dist + 1e-8)).sum(dim=1, keepdim=True)
    rel_speed = torch.norm(rel_v, dim=1, keepdim=True)
    hd = (s_ang - o_ang + _math.pi) % (2*_math.pi) - _math.pi
    heading_diff = hd.abs().unsqueeze(1)
    axis_align = torch.cos(s_ang - o_ang).unsqueeze(1)

    feat = torch.cat([dist, self_facing, other_facing, mutual,
                      approach, rel_speed, heading_diff, axis_align], dim=1)
    return feat, ['dist_to_other','self_facing_other','other_facing_self',
                  'mutual_facing','approach_speed','relative_speed',
                  'heading_diff','body_axis_align']

FEATURE_REGISTRY = {
    'skeleton': feat_skeleton,
    'motion':   feat_motion,
    'tail':     feat_tail,
    'social':   feat_social,
}

class FeatureNormalizer:
    """Feature normalizer: percentile clip + min-max → [0,1]"""
    def __init__(self, lo_pct: float = 0.1, hi_pct: float = 99.9):
        self.lo_pct = lo_pct; self.hi_pct = hi_pct
        self.lo_val = None; self.hi_val = None; self.fitted = False

    def fit(self, X: "np.ndarray | torch.Tensor") -> "FeatureNormalizer":
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        self.lo_val = np.percentile(X, self.lo_pct, axis=0).astype(np.float32)
        self.hi_val = np.percentile(X, self.hi_pct, axis=0).astype(np.float32)
        gap = self.hi_val - self.lo_val
        gap[gap < 1e-8] = 1e-6
        self.hi_val = self.lo_val + gap
        self.fitted = True
        return self

    def transform(self, X: "np.ndarray | torch.Tensor") -> "np.ndarray | torch.Tensor":
        was_tensor = isinstance(X, torch.Tensor)
        dev = X.device if was_tensor else None
        if was_tensor: X = X.cpu().numpy()
        lo, hi = self.lo_val, self.hi_val
        X_c = np.clip(X, lo, hi)
        X_n = np.clip((X_c - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        if was_tensor: X_n = torch.from_numpy(X_n).to(dev)
        return X_n

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        np.savez(path, lo=self.lo_val, hi=self.hi_val, lo_pct=self.lo_pct, hi_pct=self.hi_pct)

    def load(self, path: str):
        d = np.load(path)
        self.lo_val = d['lo']; self.hi_val = d['hi']
        self.lo_pct = float(d['lo_pct']); self.hi_pct = float(d['hi_pct'])
        self.fitted = True
        return self


class FeatureIndexer:
    """
    Automatically compute index ranges for each attribute in the feature vector,
    and maintain the list of attribute names.
    """
    def __init__(self):
        self.mp = {} # {name: slice}
        self.offset = 0
        self.feature_names = [] # Block names
        self.flat_attributes = [] # Flattened attribute names for every dimension

    def add_feature(self, name, dim, attributes=None):
        """
        Add a feature block.
        Args:
            name (str): Feature name (Block name)
            dim (int): Feature dimension
            attributes (list[str]): List of specific attribute names for this block (length should equal dim)
        """
        start = self.offset
        end = start + dim
        self.mp[name] = slice(start, end)
        self.feature_names.append(name)
        self.offset = end

        if attributes:
            if len(attributes) != dim:
                # Fallback if length mismatch
                attributes = [f"{name}_{i}" for i in range(dim)]
            self.flat_attributes.extend(attributes)
        else:
            self.flat_attributes.extend([f"{name}_{i}" for i in range(dim)])

    def get_slice(self, name):
        """Return slice object for convenient tensor[...] usage"""
        if name not in self.mp:
            raise ValueError(f"Attribute {name} not found in indexer")
        return self.mp[name]

    def get_attribute_names(self):
        """Return flattened list of all attribute names"""
        return self.flat_attributes

    def __repr__(self):
        return f"FeatureIndexer(features={self.feature_names}, total_dim={self.offset})"

class MouseTailMerger:
    def __init__(self, max_instance_num=2, distance_threshold=30.0, cost_epsilon=5.0):
        self.max_instance_num = max_instance_num
        self.distance_threshold = distance_threshold
        self.cost_epsilon = cost_epsilon  # Only used for debug/log, actual tie-breaking via ID
        self.prev_mouse_tail_ids = None  # List[Optional[int]], length = max_instance_num

    def merge_frame(self, mouse_instances, tail_instances, frame_img_shape=None):
        max_n = len(mouse_instances)
        assert len(tail_instances) == max_n, "Mouse and Tail instance lists must have same length"

        # Extract valid instances (preserving original indices)
        mouse_valid = [(i, inst) for i, inst in enumerate(mouse_instances) if inst is not None]
        tail_valid = [(j, inst) for j, inst in enumerate(tail_instances) if inst is not None]
        mouse_insts = [inst for _, inst in mouse_valid]
        tail_insts = [inst for _, inst in tail_valid]
        orig_mouse_indices = [i for i, _ in mouse_valid]

        n_mouse = len(mouse_insts)
        n_tail = len(tail_insts)

        matched_tail_for_mouse = [None] * n_mouse

        if n_mouse == 0:
            return [None] * max_n

        if n_tail > 0:
            # === 1. Build cost matrix (float32 + tiny ID tie-breaker) ===
            cost_matrix = np.full((n_mouse, n_tail), np.inf, dtype=np.float32)
            scale_w, scale_h = (frame_img_shape[1], frame_img_shape[0]) if frame_img_shape else (1.0, 1.0)

            for i, m in enumerate(mouse_insts):
                if m.points_num < 4:
                    continue
                base_px = (m.points_xy[3][0] * scale_w, m.points_xy[3][1] * scale_h)
                orig_idx = orig_mouse_indices[i]
                expected_tail_id = None
                if (self.prev_mouse_tail_ids is not None and
                    orig_idx < len(self.prev_mouse_tail_ids)):
                    expected_tail_id = self.prev_mouse_tail_ids[orig_idx]

                for j, t in enumerate(tail_insts):
                    if t.points_num < 1:
                        continue
                    tip_px = (t.points_xy[0][0] * scale_w, t.points_xy[0][1] * scale_h)
                    geo_dist = euclidean_distance(base_px, tip_px)

                    # Tie-breaker: only when ID is continuous, subtract a negligible amount
                    tie_breaker = 0.0
                    if (expected_tail_id is not None and
                        t.id is not None and
                        t.id == expected_tail_id):
                        tie_breaker = 1e-6  # Small enough to only break floating-point ties

                    cost_matrix[i, j] = geo_dist - tie_breaker

            # === 2. Hungarian matching (automatically handles all size relationships and ties) ===
            # Replace inf with large number for Hungarian algorithm
            cost_for_hungarian = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)

            # Apply matching results (filter invalid matches)
            for r, c in zip(row_ind, col_ind):
                if np.isfinite(cost_matrix[r, c]):
                    matched_tail_for_mouse[r] = tail_insts[c]

            # === 3. Special case: Mouse count decreased (but Tail not decreased) → prioritize continuity ===
            prev_active_count = sum(1 for tid in (self.prev_mouse_tail_ids or []) if tid is not None)
            if (self.prev_mouse_tail_ids is not None and
                n_mouse < prev_active_count and
                n_tail >= n_mouse):

                prev_active_tail_ids = set(tid for tid in self.prev_mouse_tail_ids if tid is not None)
                continuity_info = []
                for i in range(n_mouse):
                    tail = matched_tail_for_mouse[i]
                    has_cont = (tail is not None and
                                tail.id is not None and
                                tail.id in prev_active_tail_ids)
                    continuity_info.append((not has_cont, i))  # Continuity first

                continuity_info.sort()
                keep_indices = [idx for _, idx in continuity_info[:n_mouse]]

                new_matched = [None] * n_mouse
                for new_i, old_i in enumerate(keep_indices):
                    new_matched[new_i] = matched_tail_for_mouse[old_i]
                matched_tail_for_mouse = new_matched

        # === 4. Build output (strictly preserve original mouse index order) ===
        output_frame = [None] * max_n
        for idx_in_valid, (orig_idx, mouse_inst) in enumerate(zip(orig_mouse_indices, mouse_insts)):
            tail_inst = matched_tail_for_mouse[idx_in_valid]
            new_inst = self._merge_instance(mouse_inst, tail_inst)
            output_frame[orig_idx] = new_inst

        # === 5. Update state ===
        next_prev = [None] * max_n
        for idx_in_valid, orig_idx in enumerate(orig_mouse_indices):
            tail_inst = matched_tail_for_mouse[idx_in_valid]
            next_prev[orig_idx] = tail_inst.id if (tail_inst and tail_inst.id is not None) else None
        self.prev_mouse_tail_ids = next_prev

        return output_frame

    def _merge_instance(self, mouse_inst, tail_inst):
        """Merge a single instance"""
        mouse_points_flat = []
        for k in range(mouse_inst.points_num):
            x, y = mouse_inst.points_xy[k]
            s = mouse_inst.points_score[k]
            if x is None or y is None or s is None:
                mouse_points_flat.extend([0.0, 0.0, 0.0])
            else:
                vis_flag = 2 if abs(s - 1.0) < 1e-5 else float(s)
                mouse_points_flat.extend([float(x), float(y), vis_flag])

        tail_points_flat = []
        tail_id = None
        if tail_inst is not None:
            tail_id = tail_inst.id
            for k in range(tail_inst.points_num):
                x, y = tail_inst.points_xy[k]
                s = tail_inst.points_score[k]
                if x is None or y is None or s is None:
                    tail_points_flat.extend([0.0, 0.0, 0.0])
                else:
                    vis_flag = 2 if abs(s - 1.0) < 1e-5 else float(s)
                    tail_points_flat.extend([float(x), float(y), vis_flag])
        else:
            tail_points_flat = [0.0, 0.0, 0.0] * 3  # Tail has 3 points

        cls = int(mouse_inst.cls)
        xywh = [float(v) for v in mouse_inst.xywh]
        data_line = [cls] + xywh + mouse_points_flat + tail_points_flat

        if hasattr(mouse_inst, 'id') and mouse_inst.id is not None:
            data_line.append(int(mouse_inst.id))

        new_inst = YOLOPoseData(data_line)
        new_inst.id = [mouse_inst.id, tail_id]
        return new_inst

    def reset(self):
        """Reset state (call when starting a new video)"""
        self.prev_mouse_tail_ids = None

class MouseBehaviorDataset(Dataset):
    def __init__(self, dataset_config, label_map=None, seq_length=5, stride=5,
                 frame_interval=1, transform=None, is_train=True,
                 purity_threshold=1.0, boundary_margin=0, short_gap_max=10,
                 per_video_normalize: bool = False):  # New: per-video FeatureNormalizer for train/inference consistency
        self.dataset_config = dataset_config
        self.label_map = label_map
        self.seq_length = seq_length
        self.stride = stride
        self.frame_interval = frame_interval
        self.transform = transform
        self.purity_threshold = purity_threshold
        self.boundary_margin = boundary_margin
        self.short_gap_max = short_gap_max
        self.per_video_normalize = per_video_normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.keypoints, self.Merged_data, self.Caled_data = self.load_and_cache_keypoints()
        self.total_frames = len(self.keypoints)

        # Per-video keypoint frame counts, used by read_behavior() for per-video label alignment.
        # Derived from Merged_data (available both from cache and fresh load).
        self._per_video_frame_counts = [len(video) for video in self.Merged_data]

        if is_train:
            self.raw_labels = self.read_behavior()  # [total_frames,] or [total_frames, 1]
            if isinstance(self.raw_labels, torch.Tensor):
                self.labels = self.raw_labels.squeeze().cpu().numpy()  # Convert to numpy for ease of processing
            else:
                self.labels = np.array(self.raw_labels).squeeze()
            if self.labels is not None and len(self.labels) > 0:
                if len(self.labels) != len(self.keypoints):
                    print(f"[Alignment] Warning: labels {len(self.labels)} != keypoints {len(self.keypoints)} "
                          f"after per-video alignment. Using min as safety truncation.")
                self.total_frames = min(len(self.keypoints), len(self.labels))
        else:
            self.labels = None  # May have no labels during testing

        # Pre-compute all valid window start indices
        if self.labels is None or len(self.labels) == 0:
             self.labels = None # Ensure it is None if empty

        self.valid_start_indices = self._build_valid_windows()

    def _compute_excluded_frames(self) -> set:
        """Compute all boundary noise frames (boundary_margin frames around behavior transition points)."""
        if self.labels is None or self.boundary_margin <= 0:
            return set()
        excluded = set()
        n = len(self.labels)
        for i in range(n - 1):
            if self.labels[i] != self.labels[i + 1]:
                lo = max(0, i - self.boundary_margin + 1)
                hi = min(n - 1, i + self.boundary_margin)
                for f in range(lo, hi + 1):
                    excluded.add(f)
        return excluded

    def _build_valid_windows(self):
        """Build all window start indices that satisfy the label purity requirement"""
        if self.labels is None:
            # Test mode: no filtering, all windows are valid
            window_span = (self.seq_length - 1) * self.frame_interval + 1
            if self.total_frames < window_span:
                return []
            return list(range(0, self.total_frames - window_span + 1, self.stride))

        excluded_frames = self._compute_excluded_frames()

        valid_starts = []
        window_span = (self.seq_length - 1) * self.frame_interval + 1
        max_start = self.total_frames - window_span

        start = 0
        while start <= max_start:
            # Get all frame indices within the window
            indices = [start + i * self.frame_interval for i in range(self.seq_length)]
            if indices[-1] >= self.total_frames:
                break

            # Boundary noise filtering: skip if any frame in window falls in excluded set
            if excluded_frames and any(f in excluded_frames for f in indices):
                start += self.stride
                continue

            window_labels = self.labels[indices]

            # Skip windows containing unknown labels (-1), ensure no illegal frames in training
            if np.any(window_labels < 0):
                start += self.stride
                continue

            # Count majority class
            counter = Counter(window_labels)
            most_common_label, count = counter.most_common(1)[0]
            purity = count / len(window_labels)

            if purity >= self.purity_threshold:
                valid_starts.append((start, most_common_label))  # Store start position and final label

            start += self.stride

        return valid_starts

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        if self.labels is not None:
            start_frame, final_label = self.valid_start_indices[idx]
        else:
            start_frame = self.valid_start_indices[idx]
            final_label = -1

        indices = [start_frame + i * self.frame_interval for i in range(self.seq_length)]
        keypoint_sequence = self.keypoints[indices]

        if self.transform:
            keypoint_sequence = self.transform(keypoint_sequence)

        keypoint_sequence = torch.as_tensor(keypoint_sequence, dtype=torch.float32).to(self.device)

        # Return int label (consistent with original)
        return keypoint_sequence, int(final_label)


    def get_label_distribution(self):
        """
        Count the number and percentage of samples for each category in the entire dataset.
        Returns: dict {label: count}, dict {label: percentage}
        """
        label_counts = {}
        total_samples = len(self)

        if total_samples == 0:
            return {}, {}

        # Iterate through all samples (windows), get the label of the middle frame
        for idx in range(total_samples):
            start = idx * self.stride
            mid_idx = start + self.seq_length // 2
            label = int(self.labels[mid_idx].item()) if isinstance(self.labels, torch.Tensor) else int(
                self.labels[mid_idx])

            label_counts[label] = label_counts.get(label, 0) + 1

        # Compute ratios
        label_ratios = {label: count / total_samples for label, count in label_counts.items()}

        return label_counts, label_ratios

    def _get_config_hash(self):

        # Extract key parameters that affect data results
        config_content = {
            "Mouse_files": self.dataset_config.get("Mouse_key_point_file", []),
            "Tail_files": self.dataset_config.get("Tail_key_point_file", []),
            "max_instances": self.dataset_config.get("max_instances_num", 2),
            "method_version": "v1.4",  # v1.4: per_video_normalize support
            "per_video_normalize": getattr(self, 'per_video_normalize', False),
        }

        # Convert dict to sorted JSON string (ensure consistent ordering)
        config_str = json.dumps(config_content, sort_keys=True)

        # Compute MD5
        md5_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
        return md5_hash

    def load_and_cache_keypoints(self):
        cache_dir = ".\dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)
        config_hash = self._get_config_hash()
        cache_path = os.path.join(cache_dir, f"cached_data_{config_hash}.pt")

        if os.path.exists(cache_path):
            print(f"\n[Cache] Cache file detected, fast loading: {cache_path}")
            try:
                cache_dict = torch.load(cache_path, weights_only=False)
                print("[Cache] Loading complete!")
                if 'feature_indexer' in cache_dict:
                    self.feature_indexer = cache_dict['feature_indexer']
                if 'normalizer' in cache_dict:
                    self.normalizer = cache_dict['normalizer']
                return cache_dict['final_tensor'], cache_dict['Merged_data'], cache_dict['Caled_data']
            except Exception as e:
                print(f"[Cache] Cache file corrupted, will reprocess. Error: {e}")

        print(f"\n[Cache] No matching cache found (fingerprint: {config_hash}), starting raw text file processing...")
        Merged_data, Caled_data, final_tensor = self.read_keypoints()

        try:
            # Save as dict structure
            cache_dict = {
                'final_tensor': final_tensor,
                'Merged_data': Merged_data,
                'Caled_data': Caled_data,
                'feature_indexer': self.feature_indexer,
                'normalizer': self.normalizer,
            }
            torch.save(cache_dict, cache_path)
            # Also save independent normalization parameter file (for inference etc.)
            norm_path = os.path.join(cache_dir, f"feature_norm_{config_hash}.npz")
            self.normalizer.save(norm_path)
            print(f"[Cache] Results saved to: {cache_path}")
            print(f"[Cache] Normalization parameters saved to: {norm_path}")
        except Exception as e:
            print(f"[Cache] Failed to save cache: {e}")

        return final_tensor, Merged_data, Caled_data

    # ------------------------------------------------------------------
    # Missing value imputation (linear / Kalman)
    # ------------------------------------------------------------------
    @staticmethod
    def _impute_series(values: list, short_gap_max: int) -> list:
        """
        In-place imputation of a 1D sequence (list of float|None).

        Short gaps (<= short_gap_max) → linear interpolation
        Long gaps (> short_gap_max) → constant-velocity Kalman filter
        Invalid segments at start/end → backward/forward fill nearest valid value

        Returns new repaired list (original list unchanged).
        """
        n = len(values)
        out = list(values)

        # ---- Collect all None segments ----
        i = 0
        while i < n:
            if out[i] is None:
                j = i
                while j < n and out[j] is None:
                    j += 1
                # [i, j) all None
                left_val = out[i - 1] if i > 0 else None
                right_val = out[j] if j < n else None

                gap_len = j - i

                if left_val is None and right_val is None:
                    # Entire segment is None, cannot impute
                    i = j
                    continue

                if left_val is None:
                    # Leading segment: forward fill
                    for k in range(i, j):
                        out[k] = right_val
                elif right_val is None:
                    # Trailing segment: backward fill
                    for k in range(i, j):
                        out[k] = left_val
                elif gap_len <= short_gap_max:
                    # Short gap: linear interpolation
                    for k in range(gap_len):
                        t = (k + 1) / (gap_len + 1)
                        out[i + k] = left_val + t * (right_val - left_val)
                else:
                    # Long gap: 1D constant-velocity Kalman filter
                    # Collect several valid observations on each side for initialization
                    obs_before = []
                    for b in range(max(0, i - 5), i):
                        if out[b] is not None:
                            obs_before.append((b, out[b]))
                    obs_after = []
                    for a in range(j, min(n, j + 5)):
                        if out[a] is not None:
                            obs_after.append((a, out[a]))

                    # Forward Kalman prediction from left_val
                    vel0 = 0.0
                    if len(obs_before) >= 2:
                        dt = obs_before[-1][0] - obs_before[-2][0]
                        vel0 = (obs_before[-1][1] - obs_before[-2][1]) / max(dt, 1)

                    # State [pos, vel], simple constant-velocity model
                    x = left_val
                    v = vel0
                    Q = 1e-2   # Process noise
                    R = 1.0    # Observation noise
                    P = 1.0    # Error covariance

                    # Predict and fill gap
                    for k in range(gap_len):
                        # predict
                        x = x + v
                        P = P + Q
                        # Use right_val for update in second half
                        if k == gap_len - 1:
                            # Final step: observation update with right_val
                            K_gain = P / (P + R)
                            x = x + K_gain * (right_val - x)
                            v = 0.0
                        out[i + k] = x

                i = j
            else:
                i += 1

        return out

    def _impute_merged_data(self, merged_data, num_keypoints: int, short_gap_max: int = 10):
        """
        In-place imputation of merged_data (video × frame × slot list of YOLOPoseData|None).

        Only impute None coordinate values within existing instances:
          - inst.xywh[j]       (j=0..3)
          - inst.points_xy[k]  each coordinate component (k=0..num_keypoints-1, 0=x,1=y)
        Does not create new instances (slots that are None are not filled).
        """
        for video_frames in merged_data:          # video_frames: List[List[inst|None]]
            T = len(video_frames)
            if T == 0:
                continue
            # Infer slot count
            max_slots = max(len(frame) for frame in video_frames) if video_frames else 0

            for slot in range(max_slots):
                # xywh: 4 components
                for j in range(4):
                    series = []
                    for frame in video_frames:
                        inst = frame[slot] if slot < len(frame) else None
                        series.append(inst.xywh[j] if inst is not None else None)
                    fixed = self._impute_series(series, short_gap_max)
                    for t, frame in enumerate(video_frames):
                        inst = frame[slot] if slot < len(frame) else None
                        if inst is not None:
                            inst.xywh[j] = fixed[t]

                # points_xy: num_keypoints x 2 components
                for k in range(num_keypoints):
                    for coord in range(2):   # 0=x, 1=y
                        series = []
                        for frame in video_frames:
                            inst = frame[slot] if slot < len(frame) else None
                            if inst is None:
                                series.append(None)
                            elif k >= len(inst.points_xy) or inst.points_xy[k][coord] is None:
                                series.append(None)
                            else:
                                series.append(inst.points_xy[k][coord])
                        fixed = self._impute_series(series, short_gap_max)
                        for t, frame in enumerate(video_frames):
                            inst = frame[slot] if slot < len(frame) else None
                            if inst is not None and k < len(inst.points_xy):
                                inst.points_xy[k][coord] = fixed[t]

    def read_keypoints(self):
        Mouse_key_point_file = self.dataset_config["Mouse_key_point_file"]
        Tail_key_point_file = self.dataset_config["Tail_key_point_file"]

        max_instance_num = self.dataset_config["max_instances_num"]

        num_keypoints = 10  # 7 mouse + 3 tail
        # selected_frame = self.dataset_config["selected_frame"]
        # vidio_path = self.dataset_config["vidio_path"]

        Mouse_data = []
        Tail_data = []
        all_tensor_list = []
        num_workers = max(1, min(8, len(Mouse_key_point_file)))  # At least 1
        # === Parallel processing of Mouse files ===
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            mouse_futures = [
                executor.submit(process_pose_file, file_, max_instance_num, is_mouse=True)
                for file_ in Mouse_key_point_file
            ]
            Mouse_data = [f.result() for f in mouse_futures]

        # === Parallel processing of Tail files ===
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            tail_futures = [
                executor.submit(process_pose_file, file_, max_instance_num, is_mouse=False)
                for file_ in Tail_key_point_file
            ]
            Tail_data = [f.result() for f in tail_futures]


        assert len(Mouse_data) == len(Tail_data), "Number of files must match"

        Merged_data = self.merge_mouse_tail_datasets(Mouse_data, Tail_data)

        # Missing keypoint imputation (short gaps: linear interpolation, long gaps: Kalman filter)
        short_gap_max = getattr(self, "short_gap_max", 10)
        self._impute_merged_data(Merged_data, num_keypoints=num_keypoints, short_gap_max=short_gap_max)

        # Caled_data = self.cal_feature(Merged_data, num_keypoints)
        sorted_data = Merged_data


        final_tensor= self.build_centered_tensors_concatenated(# Assemble large vector
                            sorted_data,
                            max_instance_num=2,
                            num_keypoints=10,  # 7 mouse + 3 tail
                            include_absolute_box_xy=True,
                            per_video_normalize=self.per_video_normalize,
                        )  # shape: [N, D]

        # Feature normalization: percentile clip + min-max → [0,1]
        # When per_video_normalize=True, the above call already normalizes
        # each video independently — skip the global pass to avoid double-normalizing.
        if not self.per_video_normalize:
            self.normalizer = FeatureNormalizer()
            final_tensor = self.normalizer.fit_transform(final_tensor)
        else:
            # normalizer already set inside build_centered_tensors_concatenated
            pass
        return Merged_data, sorted_data, final_tensor


    def read_behavior(self):

        self.behavior_index_map = {k: int(v) for k, v in self.label_map.items()}  # value to idx
        # Handle label_merge: merged source classes (e.g. climbsocial) are not in label_map,
        # but annotation files may still use the original name, need to map to target class ID
        from pathlib import Path as _BM_Path
        import json as _BM_Json
        _ds_json_path = _BM_Path("config/dataset_config.json")
        if _ds_json_path.exists():
            with open(_ds_json_path, "r", encoding="utf-8") as _f:
                _ds_json = _BM_Json.load(_f)
            _merge_cfg = _ds_json.get("label_merge", {})
            if _merge_cfg.get("enabled", False):
                for _g in _merge_cfg.get("groups", []):
                    _target = _g.get("target", "")
                    if _target in self.behavior_index_map:
                        _tid = self.behavior_index_map[_target]
                        for _src in _g.get("sources", []):
                            self.behavior_index_map[_src] = _tid
        self.behavior_index_map_turn = {int(v):k  for k, v in self.label_map.items()}  # idx to value
        max_instance_num = self.dataset_config["max_instances_num"]

        behavior_file_mouse1 = self.dataset_config["behavior_file_mouse1"]
        behavior_file_mouse2 = self.dataset_config["behavior_file_mouse2"]

        final_tensor_list = []

        def _load_single_behavior_file(file_, kp_frame_count):
            """
            Load behavior annotations for a single video, return a label tensor fully aligned
            with the video's keypoint frame count, plus an alignment report.

            Alignment rules (done within each video, already aligned before concatenation):
              - label coverage < kp_frame_count: pad remaining frames with -1
              - label coverage > kp_frame_count: clip annotation end_frame to kp_frame_count, discard overflow
              - illegal/unknown behavior types: corresponding frames remain -1, not used in training
            Returns: (label_tensor [kp_frame_count, 1], report_dict)
            """
            annotations = []
            with open(file_, 'r') as f:
                lines = f.readlines()
                record_flag = False
                for line in lines:
                    stripped_line = line.strip()
                    if "Configuration file:" in stripped_line or stripped_line == '' or '----' in stripped_line:
                        continue
                    if "S1:" in stripped_line:
                        record_flag = True
                        continue
                    if record_flag:
                        parts = stripped_line.split()
                        if len(parts) < 3:
                            continue
                        try:
                            start_frame = int(parts[0])
                            end_frame = int(parts[1])
                            behavior_type = parts[2]
                            annotations.append((start_frame, end_frame, behavior_type))
                        except ValueError:
                            pass

            # Maximum frame covered by labels (1-based end_frame)
            label_max_frame = max((a[1] for a in annotations), default=0)

            report = {
                "kp_frames": kp_frame_count,
                "label_max_frame": label_max_frame,
                "clipped_frames": max(0, label_max_frame - kp_frame_count),  # Frames where label exceeds kp
                "padded_frames": max(0, kp_frame_count - label_max_frame),   # Frames where kp exceeds label, padded with -1
                "unknown_segments": [],   # [(start1, end1, type), ...]
                "status": "ok",
            }
            if report["clipped_frames"] > 0:
                report["status"] = "label_trimmed"
            elif report["padded_frames"] > 0:
                report["status"] = "kp_padded"

            # Label tensor size is strictly kp_frame_count, initialized to -1
            label_tensor = torch.full((kp_frame_count, 1), -1, dtype=torch.long)

            for start_frame_, end_frame_, behavior_type in annotations:
                s = max(0, start_frame_ - 1)          # Convert to 0-based
                e = min(kp_frame_count, end_frame_)    # Clip to kp range
                if s >= e:
                    continue
                if behavior_type in self.behavior_index_map:
                    label_tensor[s:e, 0] = self.behavior_index_map[behavior_type]
                else:
                    # Illegal behavior type: corresponding frames remain -1
                    report["unknown_segments"].append((s + 1, e, behavior_type))
                    if report["status"] == "ok":
                        report["status"] = "has_unknown"

            return label_tensor, report

        def _print_alignment_report(video_name, report, mouse_tag):
            kp = report["kp_frames"]
            lm = report["label_max_frame"]
            status = report["status"]

            if status == "ok":
                print(f"  [OK]    {mouse_tag} {video_name}: {kp} frames fully aligned")
            elif status == "label_trimmed":
                print(f"  [TRIM]  {mouse_tag} {video_name}: label {lm} frames > kp {kp} frames, "
                      f"clipped {report['clipped_frames']} tail frames from label")
            elif status == "kp_padded":
                print(f"  [PAD]   {mouse_tag} {video_name}: kp {kp} frames > label {lm} frames, "
                      f"last {report['padded_frames']} frames padded with -1")
            elif status == "has_unknown":
                segs = ", ".join(f"[{s}-{e}]{t}" for s, e, t in report["unknown_segments"])
                print(f"  [UNK]   {mouse_tag} {video_name}: {kp} frames aligned, "
                      f"illegal behavior segments → -1: {segs}")

        # ---- Summary statistics ----
        total_videos = 0
        ok_count = 0
        trim_count = 0
        pad_count = 0
        unk_count = 0

        print("\n[Alignment] ===== Behavior Label Alignment Check =====")

        all_tensor_list = []
        if behavior_file_mouse1:
            for idx, file_ in enumerate(behavior_file_mouse1):
                kp_count = self._per_video_frame_counts[idx] if idx < len(self._per_video_frame_counts) else 0
                label_tensor, report = _load_single_behavior_file(file_, kp_count)
                video_name = Path(file_).stem
                _print_alignment_report(video_name, report, "Mouse1")
                total_videos += 1
                st = report["status"]
                if st == "ok": ok_count += 1
                elif st == "label_trimmed": trim_count += 1
                elif st == "kp_padded": pad_count += 1
                else: unk_count += 1
                all_tensor_list.append(label_tensor)
            if all_tensor_list:
                final_tensor = torch.cat(all_tensor_list, dim=0)
                final_tensor_list.extend(final_tensor)

        all_tensor_list = []
        if behavior_file_mouse2:
            for idx, file_ in enumerate(behavior_file_mouse2):
                kp_count = self._per_video_frame_counts[idx] if idx < len(self._per_video_frame_counts) else 0
                label_tensor, report = _load_single_behavior_file(file_, kp_count)
                video_name = Path(file_).stem
                _print_alignment_report(video_name, report, "Mouse2")
                total_videos += 1
                st = report["status"]
                if st == "ok": ok_count += 1
                elif st == "label_trimmed": trim_count += 1
                elif st == "kp_padded": pad_count += 1
                else: unk_count += 1
                all_tensor_list.append(label_tensor)
            if all_tensor_list:
                final_tensor = torch.cat(all_tensor_list, dim=0)
                final_tensor_list.extend(final_tensor)

        print(f"\n[Alignment] {total_videos} videos processed through alignment:")
        print(f"  Fully aligned: {ok_count} | "
              f"Label trimmed: {trim_count} | "
              f"Kp tail padded with -1: {pad_count} | "
              f"Contains illegal labels: {unk_count}")
        print("[Alignment] =====================================\n")

        if final_tensor_list:
            final_tensor = torch.cat(final_tensor_list, dim=0)
            return final_tensor
        else:
            return torch.empty(0, 1, dtype=torch.long)
    def build_centered_tensors_concatenated(
            self,
            sorted_data,
            max_instance_num: int = 2,
            num_keypoints: int = 10,  # 7 mouse + 3 tail
            fps: int = 30,
            feature_list: list= None,
            **kwargs
    ) -> torch.Tensor:
        """
        Refactored to use vectorized operations with a configurable feature list.

        When per_video_normalize=True (via kwargs), each video's features are
        independently percentile-clipped + min-max scaled to [0,1] using that
        video's own statistics.  This makes training and single-video inference
        numerically consistent — no normalizer needs to be saved or loaded.
        """
        if feature_list is None:
             feature_list = ['skeleton', 'motion', 'tail', 'social']

        per_video_normalize = bool(kwargs.get('per_video_normalize', False))

        # Config map for FeatureIndexer
        self.feature_indexer = FeatureIndexer()
        indexer_built = False

        # Pre-compute contexts for all videos so we can loop center_id first,
        # matching the label ordering from read_behavior (all mouse1 then all mouse2).
        video_contexts = []
        for video_data in sorted_data:
            if not video_data:
                continue

            # 1. Raw Data to Tensor
            points_t, box_t, mask_t = video_to_tensor(video_data, max_instance_num, num_keypoints, device=self.device)
            T = points_t.shape[0]
            if T == 0:
                continue

            # 1.5 Ensure slot 0 corresponds to the left mouse (annotation convention: left=mouse1)
            # Check first frame: if both mice exist and slot 0 box_cx > slot 1, then slot 0 is the right mouse, swap
            _check_frame = 0
            while _check_frame < T and not (mask_t[_check_frame, 0] and mask_t[_check_frame, 1]):
                _check_frame += 1
            if _check_frame < T and box_t[_check_frame, 0, 0] > box_t[_check_frame, 1, 0]:
                points_t = points_t[:, [1, 0], :, :]
                box_t    = box_t[:,    [1, 0], :]
                mask_t   = mask_t[:,   [1, 0]]

            # 2. Context
            ctx = VectorizedContext(points_t, box_t, mask_t, fps, num_keypoints, self.device)
            video_contexts.append(ctx)

        if per_video_normalize:
            # ── Per-video normalization path ──────────────────────────
            n_videos = len(video_contexts)
            print(f"[FeatureNorm] Per-video normalization enabled "
                  f"({n_videos} videos, each independently scaled to [0,1])")
            # Build features for all center_ids within each video,
            # normalize with that video's own FeatureNormalizer,
            # then re-interleave to preserve mouse1-first label order.
            m1_parts: list = []   # mouse1 (center_id=0) features, per video
            m2_parts: list = []   # mouse2 (center_id=1) features, per video
            # Track per-video frame counts for downstream per-video window building
            _video_m1_lengths: list = []
            _video_m2_lengths: list = []

            for ctx in video_contexts:
                video_parts = []
                for center_id in range(max_instance_num):
                    features_for_center = []
                    for feat_name in feature_list:
                        if feat_name not in FEATURE_REGISTRY:
                            print(f"Warning: Feature '{feat_name}' not found in registry. Skipping.")
                            continue
                        calc_func = FEATURE_REGISTRY[feat_name]
                        feat_tensor, attr_names = calc_func(ctx, center_id)
                        features_for_center.append(feat_tensor)
                        if not indexer_built and center_id == 0:
                            self.feature_indexer.add_feature(feat_name, feat_tensor.shape[1], attr_names)
                    if features_for_center:
                        video_parts.append(torch.cat(features_for_center, dim=1))
                    if not indexer_built and center_id == 0:
                        indexer_built = True

                # video_parts = [m1_feat [T, D], m2_feat [T, D]]  (or just [m1_feat] if max_instance_num=1)
                T_v = video_parts[0].shape[0]
                video_cat = torch.cat(video_parts, dim=0)         # [T_v * M, D]
                # Per-video normalize
                vn = FeatureNormalizer()
                video_cat = vn.fit_transform(video_cat)
                # Split back into mouse1 / mouse2 and store
                m1_parts.append(video_cat[:T_v])
                _video_m1_lengths.append(T_v)
                if max_instance_num > 1 and len(video_parts) > 1:
                    m2_parts.append(video_cat[T_v:])
                    _video_m2_lengths.append(video_cat[T_v:].shape[0])

            # Store per-video frame counts so downstream window building
            # can restrict each centred window to its own video.
            self._video_m1_lengths = _video_m1_lengths
            self._video_m2_lengths = _video_m2_lengths

            # Re-interleave to mouse1-first → mouse2 order
            all_tensors = m1_parts + m2_parts
            if not all_tensors:
                return torch.empty(0, 0)
            result = torch.cat(all_tensors, dim=0)

            # Also fit a "dummy" normalizer on the full result for cache compatibility
            self.normalizer = FeatureNormalizer()
            self.normalizer.fit(result.cpu().numpy())
            return result

        # ── Original path (global normalization, default) ─────────────
        all_tensors = []

        # 3. Outer loop: center_id — produces [v0c0, v1c0, ..., v0c1, v1c1, ...]
        #    which aligns with read_behavior's label order (all mouse1 then all mouse2).
        for center_id in range(max_instance_num):
            for ctx in video_contexts:
                features_for_center = []

                for feat_name in feature_list:
                    if feat_name not in FEATURE_REGISTRY:
                        print(f"Warning: Feature '{feat_name}' not found in registry. Skipping.")
                        continue

                    calc_func = FEATURE_REGISTRY[feat_name]
                    feat_tensor, attr_names = calc_func(ctx, center_id)

                    features_for_center.append(feat_tensor)

                    if not indexer_built and center_id == 0:
                        self.feature_indexer.add_feature(feat_name, feat_tensor.shape[1], attr_names)

                # Concat all for this center
                if features_for_center:
                    center_tensor = torch.cat(features_for_center, dim=1)  # [T, Total_Dim]
                    all_tensors.append(center_tensor)

                if not indexer_built and center_id == 0:
                    indexer_built = True

        if not all_tensors:
            return torch.empty(0, 0)

        return torch.cat(all_tensors, dim=0)

    def get_message_in_mdata(self, x):

        max_instance_num = self.dataset_config["max_instances_num"]
        all_instances_xlist = []

        for instance_idx in range(max_instance_num):
            instance_xlist = []
            for frame in self.sorted_keypoints:
                if frame[instance_idx] is not None:
                    instance_xlist.append(getattr(frame[instance_idx], x))
                else: instance_xlist.append(None)
            all_instances_xlist.append(instance_xlist)
        return all_instances_xlist

    def merge_single_video(self, args):
        """
        Internal helper: merge mouse and tail data for a single video.
        """
        mouse_file, tail_file, max_instance_num, shape, cost_epsilon = args
        assert len(mouse_file) == len(tail_file), "Frame count mismatch in a video"

        # Create independent Merger per video (no shared state across videos)
        merger = MouseTailMerger(max_instance_num=max_instance_num, cost_epsilon=cost_epsilon)
        merged_file = []

        for frame_idx in range(len(mouse_file)):
            merged_frame = merger.merge_frame(
                mouse_instances=mouse_file[frame_idx],
                tail_instances=tail_file[frame_idx],
                frame_img_shape=shape
            )
            merged_file.append(merged_frame)

        return merged_file

    def merge_mouse_tail_datasets(self, Mouse_data, Tail_data, video_shapes=None):
        """
        Merge Mouse and Tail datasets with frame-to-frame ID continuity, processing each video in parallel.

        Args:
            Mouse_data: List[List[List[YOLOPoseData | None]]]  # [video][frame][slot]
            Tail_data:  List[List[List[YOLOPoseData | None]]]
            video_shapes: Optional[List[Tuple[int, int]]], (height, width) for each video

        Returns:
            Merged_data: same structure as input
        """
        assert len(Mouse_data) == len(Tail_data), "Number of mouse and tail files must match"
        num_videos = len(Mouse_data)
        if video_shapes is not None:
            assert len(video_shapes) == num_videos, "video_shapes length mismatch"

        max_instance_num = self.dataset_config["max_instances_num"]
        cost_epsilon = 5.0  # Consistent with original code

        # Build argument list: one entry per video
        merge_args = []
        for i in range(num_videos):
            shape = video_shapes[i] if video_shapes is not None else None
            merge_args.append((Mouse_data[i], Tail_data[i], max_instance_num, shape, cost_epsilon))

        # Merge each video in parallel
        num_workers = max(1, min(8, num_videos))  # At least 1
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            Merged_data = list(executor.map(self.merge_single_video, merge_args))

        return Merged_data

class MouseKeypointDataset(Dataset):

    def __init__(self, key_point_files, max_instance_num, data_type, img_w=1280, img_h=720):
        self.key_point_files = key_point_files
        self.max_instance_num = max_instance_num
        self.data_type = data_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_w = img_w
        self.img_h = img_h
        self.keypoints = self.read_keypoints()


    def read_keypoints(self):
        key_point_files = self.key_point_files
        max_instance_num = self.max_instance_num
        data_type = self.data_type
        if data_type not in ["YOLO","SLEAP"]:
            data_type = "YOLO"

        all_data_list = []

        if data_type == "YOLO":
            for idx, file_ in enumerate(key_point_files):

                org_data = load_pose_data_by_file_order(file_)

                sorted_data = []
                all_frame_ids = []

                # Initialize previous frame ID list (length = max_instance_num, all None initially)
                last_frame_ids = [None] * max_instance_num

                for frame_idx, frame in enumerate(org_data):
                    # Convert all instances in current frame to YOLOPoseData objects
                    current_instances = [YOLOPoseData(inst) for inst in frame]  # Each inst is a 17-element list
                    current_ids = [inst.id for inst in current_instances]

                    # Initialize output slots for current frame (max_instance_num slots)
                    new_frame = [None] * max_instance_num
                    match = [False] * max_instance_num  # Mark which slots are occupied

                    # Step 1: Try to match current instances to existing ID slots from previous frame
                    for inst in current_instances:
                        matched = False
                        for slot_idx, last_id in enumerate(last_frame_ids):
                            if last_id is not None and inst.id == last_id:
                                # Found matching slot
                                new_frame[slot_idx] = inst
                                match[slot_idx] = True
                                matched = True
                                break
                        if matched:
                            continue  # Already assigned, skip

                        # Step 2: If no match found, try first free slot
                        for slot_idx in range(max_instance_num):
                            if not match[slot_idx]:  # Slot is free
                                new_frame[slot_idx] = inst
                                match[slot_idx] = True
                                break


                    # Step 3: Build current frame ID list (for next iteration)
                    current_frame_ids = []
                    for inst in new_frame:
                        current_frame_ids.append(inst.id if inst is not None else None)

                    # Update state
                    last_frame_ids = current_frame_ids
                    sorted_data.append(new_frame)
                    all_frame_ids.append(current_frame_ids)


                all_data_list.append(sorted_data)

        elif data_type == "SLEAP":

            for idx, file_ in enumerate(key_point_files):
                df = pd.read_csv(file_)
                frame_data_list = []  # Final result: list of YOLOPoseData per frame

                # Get all columns except frame_idx
                pose_columns = [col for col in df.columns if col != 'frame_idx']
                n_points_per_mouse = 6  # snout, head_center, body_center, tailbase
                n_vals_per_mouse = n_points_per_mouse * 3  # x, y, score

                for idx, row in df.iterrows():
                    mouse_data_list = []  # All valid mouse YOLOPoseData in this frame

                    # Extract pose data (in order)
                    pose_vals = row[pose_columns].values  # shape: (N,)

                    # Attempt to split into mouse chunks
                    num_complete_mice = len(pose_vals) // n_vals_per_mouse
                    remainder = len(pose_vals) % n_vals_per_mouse
                    if remainder != 0:
                        pass  # Ignore incomplete mice (optional: pad with nan)

                    for m in range(num_complete_mice):
                        start = m * n_vals_per_mouse
                        mouse_chunk = pose_vals[start:start + n_vals_per_mouse]

                        # Do not skip entire mouse due to nan; keep and normalize (nan stays nan)
                        normalized_points = []
                        for i in range(0, len(mouse_chunk), 3):
                            x_abs = mouse_chunk[i]
                            y_abs = mouse_chunk[i + 1]
                            score = mouse_chunk[i + 2]

                            # If x or y is nan, treat entire point as missing (score also nan)
                            if pd.isna(x_abs) or pd.isna(y_abs):
                                # Keep three nan values to indicate missing keypoint
                                normalized_points.extend([np.nan, np.nan, np.nan])
                            else:
                                # x, y valid, normalize; score may be 0 (valid but unreliable), keep original
                                x_norm = float(x_abs) / self.img_w
                                y_norm = float(y_abs) / self.img_h
                                s_val = float(score) if not pd.isna(score) else np.nan
                                normalized_points.extend([x_norm, y_norm, s_val])

                        # Construct data_line: [cls, x, y, w, h, points..., id]
                        data_line = [None] + [None] * 4 + normalized_points + [None]
                        try:
                            yolo_data = YOLOPoseData(data_line)
                            mouse_data_list.append(yolo_data)
                        except Exception as e:
                            print(f"Error at frame {row['frame_idx']}, mouse {m}: {e}")
                            continue

                    frame_data_list.append(mouse_data_list)
                all_data_list.append(frame_data_list)

        return all_data_list

if __name__ == '__main__':

    dataset_config = {}

    # Fill in your own dataset paths (see config/dataset_config.example.json)
    dataset_config["Mouse_key_point_file"] = ["path/to/mouse_keypoint_dir"]
    dataset_config["Tail_key_point_file"] = ["path/to/tail_keypoint_dir"]

    dataset_config["max_instances_num"] = 2
    dataset_config["analyze_idx"] = 1
    dataset_config["selected_frame"] = 1

    dataset_config["selected_frame"] =  "0.0,1.0"

    label_map = {
    "explore_object" : 0,
    "climb" : 0,
    "self_grooming" : 0,
    "stand" : 0,
    "blank": 0,

    "positive_sniffs" : 1,
    "approach" : 1,
    "climbsocial" : 1
    }

    MBD = MouseBehaviorDataset(dataset_config, label_map=label_map, seq_length=5, stride=5,
                 frame_interval=1, transform=None, is_train=False,
                 purity_threshold=1.0)
    Merged_data = MBD.Merged_data
