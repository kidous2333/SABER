"""
InferenceWorker — runs model inference (Pose / Behavior Prediction).

Pose mode: YOLO predict on images → keypoint output.
Behavior Prediction mode: SABER 3-stage pipeline → CSV timeline.
"""

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.utils.logging_handler import ModuleLogRedirector


class InferenceWorker(BaseWorker):
    """Worker that runs model inference (Pose / Behavior Prediction)."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        redirector = None
        try:
            mode = self._params.get("mode", "Behavior Prediction")
            self.log("=" * 50, 20)
            self.log(f"Inference Starting — Mode: {mode}", 20)
            self.log("=" * 50, 20)

            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))
            import os as _os
            _os.chdir(str(project_root))

            # ---- Install redirector ----
            redirector = ModuleLogRedirector()
            redirector.log_signal.connect(self._on_worker_log)
            redirector.install()

            if mode == "Pose":
                if self._params.get("live_camera", False):
                    self._run_live_pose()
                else:
                    self._run_pose_inference()
            elif mode == "Behavior Prediction":
                self._run_behavior_inference()
            elif mode == "End-to-End":
                if self._params.get("live_camera", False):
                    self._run_live_end_to_end()
                else:
                    self._run_end_to_end()
            else:
                self.error.emit(f"Unknown inference mode: {mode}")
                self.finished.emit()
                return

            self.set_progress(100, "Inference complete")
            self.log("Inference complete.", 20)

        except Exception as e:
            self.log(f"Inference failed: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            if redirector:
                redirector.uninstall()
            self.finished.emit()

    # ── Pose inference ──────────────────────────────────────────────

    # ── Per-instance color map (BGR for OpenCV) ──
    # Mouse 1 = Red, Mouse 2 = Blue, Mouse 3 = Green, Mouse 4 = Cyan
    _INSTANCE_COLORS = [
        (0, 0, 255),     # Red    — Mouse 1
        (255, 0, 0),     # Blue   — Mouse 2
        (0, 255, 0),     # Green  — Mouse 3
        (255, 255, 0),   # Cyan   — Mouse 4
    ]

    # Skeleton edges: (from_kpt_idx, to_kpt_idx) for 6-keypoint mouse pose
    # 0=snout, 1=head_center, 2=body_center, 3=tailbase, 4=ear_left, 5=ear_right
    _SKELETON = [(0, 1), (1, 2), (2, 3), (1, 4), (1, 5)]

    @staticmethod
    def _draw_pose_frame(img, instances: list, frame_number: int = 0,
                         conf_threshold: float = 0.3, fps: float = 0.0,
                         real_fps: float = 0.0):
        """Draw keypoints + skeleton + tail + mouse label + frame counter + legend.

        Parameters
        ----------
        img : np.ndarray  (H, W, 3) BGR image
        instances : list of dicts — see per-instance keys below
        frame_number : int — current frame index
        conf_threshold : float — minimum confidence to draw a keypoint
        fps : float — EMA instant FPS (0 = hide)
        real_fps : float — real throughput FPS (0 = hide)
        """
        import cv2
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Draw each instance ──
        for inst in instances:
            label = inst["label"]
            kps = np.asarray(inst["keypoints"])
            conf = np.asarray(inst["conf"])
            color = inst["color"]
            tail_kps = inst.get("tail_kps")
            tail_conf = inst.get("tail_conf")

            # Draw mouse body skeleton
            for a, b in InferenceWorker._SKELETON:
                if a >= len(kps) or b >= len(kps):
                    continue
                if conf[a] < conf_threshold or conf[b] < conf_threshold:
                    continue
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)

            # Draw mouse body keypoints
            for k in range(len(kps)):
                if conf[k] < conf_threshold:
                    continue
                px, py = int(kps[k][0]), int(kps[k][1])
                cv2.circle(img, (px, py), 5, color, -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw tail keypoints + chain
            if tail_kps is not None and len(tail_kps) > 0:
                t_kps = np.asarray(tail_kps)
                t_conf = np.asarray(tail_conf) if tail_conf is not None else np.ones(len(t_kps))
                # Connect tailbase (mouse kp 3) to tail_start (tail kp 0)
                if len(kps) > 3 and conf[3] >= conf_threshold and t_conf[0] >= conf_threshold:
                    cv2.line(img,
                             (int(kps[3][0]), int(kps[3][1])),
                             (int(t_kps[0][0]), int(t_kps[0][1])),
                             color, 1, cv2.LINE_AA)
                # Draw tail chain: kp0→kp1→kp2→...
                for ti in range(len(t_kps) - 1):
                    if t_conf[ti] >= conf_threshold and t_conf[ti + 1] >= conf_threshold:
                        cv2.line(img,
                                 (int(t_kps[ti][0]), int(t_kps[ti][1])),
                                 (int(t_kps[ti + 1][0]), int(t_kps[ti + 1][1])),
                                 color, 1, cv2.LINE_AA)
                # Draw tail keypoints (smaller circles to distinguish from body)
                for ti in range(len(t_kps)):
                    if t_conf[ti] < conf_threshold:
                        continue
                    px, py = int(t_kps[ti][0]), int(t_kps[ti][1])
                    cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw mouse label near the first valid keypoint
            label_pt = None
            for k in range(len(kps)):
                if conf[k] >= conf_threshold:
                    label_pt = (int(kps[k][0]) + 10, int(kps[k][1]) - 10)
                    break
            if label_pt is None and len(kps) > 0:
                label_pt = (int(kps[0][0]) + 10, int(kps[0][1]) - 10)
            if label_pt is not None:
                cv2.putText(img, label, label_pt, font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(img, label, label_pt, font, 0.5, color, 2, cv2.LINE_AA)

        # ── Frame counter + FPS (top-left) ──
        cv2.putText(img, f"Frame: {frame_number}", (10, 30), font,
                    0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"Frame: {frame_number}", (10, 30), font,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
        if real_fps > 0:
            fps_text = f"FPS: {real_fps:.0f}"
            cv2.putText(img, fps_text, (10, 60), font,
                        0.75, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, fps_text, (10, 60), font,
                        0.75, (0, 0, 0), 2, cv2.LINE_AA)

        # ── Color legend (top-right) ──
        legend_x = w - 170
        legend_y = 20
        legend_rect_size = 18
        for inst in instances:
            color = inst["color"]
            label = inst["label"]
            cv2.rectangle(img, (legend_x, legend_y),
                          (legend_x + legend_rect_size, legend_y + legend_rect_size),
                          color, -1)
            cv2.rectangle(img, (legend_x, legend_y),
                          (legend_x + legend_rect_size, legend_y + legend_rect_size),
                          (255, 255, 255), 2)
            cv2.putText(img, label, (legend_x + 24, legend_y + 16), font,
                        0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, label, (legend_x + 24, legend_y + 16), font,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)
            legend_y += 24

    def _run_pose_inference(self):
        """Run YOLO pose inference with tracking for videos + custom rendering."""
        import cv2, json as _json

        pose_run_dir = self._params.get("pose_run_dir", "")
        sources = self._params.get("pose_inference_sources", [])
        output_path = self._params.get("output_path", "") or "results.csv"

        if not pose_run_dir:
            self.error.emit("Pose Run Dir not specified")
            return
        sources = [s for s in sources if s and str(s).strip()]
        if not sources:
            self.error.emit("Images/Videos not specified")
            return

        run_path = Path(pose_run_dir)
        weights = run_path / "weights" / "best.pt"
        if not weights.exists():
            alt = list(run_path.glob("*.pt"))
            weights = alt[0] if alt else weights
        if not weights.exists():
            self.error.emit(f"Pose weights not found: {weights}")
            return

        self.set_progress(10, "Loading YOLO pose model...")
        self.log(f"Pose model: {weights.name}", 20)

        from src.pose_trainer import _ensure_tmp_registered
        _ensure_tmp_registered()
        from ultralytics import YOLO
        model = YOLO(str(weights))

        # ── Optional tail model ──
        tail_model = None
        if bool(self._params.get("tail_run_dir", "") and str(self._params.get("tail_run_dir", "")).strip()):
            tail_run_dir = self._params.get("tail_run_dir", "")
            if tail_run_dir and str(tail_run_dir).strip():
                tail_weights = Path(tail_run_dir) / "weights" / "best.pt"
                if tail_weights.exists():
                    tail_model = YOLO(str(tail_weights))
                    self.log("Two-model mode: body + tail in parallel", 20)
                else:
                    self.log(f"Tail weights not found: {tail_weights}", 30)

        self.set_progress(10, "Counting total frames...")
        self.log(f"Sources: {len(sources)} item(s)", 20)

        # ── Visualization output directory ──
        save_viz = self._params.get("save_visualization", True)
        viz_output_dir = self._params.get("viz_output_dir", "")
        if viz_output_dir and str(viz_output_dir).strip():
            out_dir = Path(viz_output_dir)
        else:
            out_dir = Path(output_path).parent / "pose_inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

        # ── Count total frames across all sources ──
        total_frames = 0
        source_total_frames = {}  # source_index → count
        for i, src in enumerate(sources):
            sp = Path(src)
            if not sp.exists():
                continue
            if sp.suffix.lower() in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(sp))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                total_frames += max(n, 1)
                source_total_frames[i] = max(n, 1)
                self.log(f"  [{i+1}] Video: {sp.name} — ~{max(n, 1)} frames", 20)
            elif sp.is_dir():
                n = sum(1 for f in sp.iterdir() if f.suffix.lower() in IMG_EXTS)
                total_frames += max(n, 1)
                source_total_frames[i] = max(n, 1)
                self.log(f"  [{i+1}] Folder: {sp.name} — {max(n, 1)} images", 20)
            else:
                source_total_frames[i] = 1
                total_frames += 1

        if total_frames == 0:
            self.error.emit("No frames found in any source")
            return
        self.log(f"Total frames to process: {total_frames}", 20)

        # Classify sources
        video_sources = set()
        for i, src in enumerate(sources):
            if Path(src).suffix.lower() in VIDEO_EXTS:
                video_sources.add(i)

        n_label_files = 0
        processed_frames = 0
        source_results = []  # list of per-source dicts

        for i, src in enumerate(sources):
            src_path = Path(src)
            if self._cancelled:
                self.log("  Cancelled by user before processing next source", 30)
                break
            if not src_path.exists():
                self.log(f"  Skipping missing: {src}", 30)
                continue
            self.log(f"  [{i+1}/{len(sources)}] {src_path.name}", 20)

            src_out = out_dir / f"source_{i}"
            src_out.mkdir(parents=True, exist_ok=True)
            frames_dir = src_out / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            labels_dir = src_out / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            is_video = i in video_sources

            src_frames = []
            src_tracking_json = None

            if is_video:
                import time as _time
                from queue import Queue as _Queue
                from threading import Thread as _Thread

                if tail_model is not None:
                    # ── Two-model mode: parallel CUDA streams ──
                    self.log("  Two-model parallel tracking (CUDA streams)...", 20)
                    # Warmup both models
                    _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(_dummy, verbose=False, imgsz=640)
                    tail_model.predict(_dummy, verbose=False, imgsz=640)

                    tail_queue = _Queue(maxsize=256)
                    def _tail_worker():
                        try:
                            for _r in tail_model.track(
                                source=str(src_path), persist=True, stream=True,
                                save=False, save_txt=False, save_frames=False,
                                project=str(out_dir), name=f"source_{i}_tail",
                                exist_ok=True, verbose=False, imgsz=640, batch=16,
                            ):
                                tail_queue.put(_r)
                        except BaseException as _e:
                            tail_queue.put(RuntimeError(f"Tail: {_e}"))
                        tail_queue.put(None)
                    tail_thread = _Thread(target=_tail_worker, daemon=True)
                    tail_thread.start()

                    results = model.track(
                        source=str(src_path), persist=True, stream=True,
                        save=False, save_txt=False, save_frames=False,
                        project=str(out_dir), name=f"source_{i}",
                        exist_ok=True, verbose=False, imgsz=640, batch=16,
                    )
                else:
                    # ── Single model tracking (combined or body-only) ──
                    self.log("  Single-model tracking (batch=16)...", 20)
                    tail_queue = None
                    tail_thread = None
                    results = model.track(
                        source=str(src_path), persist=True, stream=True,
                        save=False, save_txt=False, save_frames=False,
                        project=str(out_dir), name=f"source_{i}",
                        exist_ok=True, verbose=False, imgsz=640, batch=16,
                    )

                tracking_data = {}
                frame_count = 0
                _last_frame_time = _time.time()
                _fps_ema = 0.0
                _fps_alpha = 0.1

                _t = {"model": 0.0, "post": 0.0}
                _t_start = _time.time()
                _prev_profile_frame = 0
                _prev_profile_time = _t_start
                _interval_fps = 0.0

                max_mice = int(self._params.get("max_mice", 2))
                slot_ids = [None] * max_mice
                identity_initialized = False

                _results_iter = iter(results)
                frame_idx = 0
                while True:
                    _t_gpu = _time.time()
                    try:
                        result = next(_results_iter)
                    except StopIteration:
                        break
                    _t["model"] += _time.time() - _t_gpu
                    if self._cancelled:
                        self.log("  Cancelled by user", 30)
                        break
                    if result.boxes is None or len(result.boxes) == 0:
                        # Save empty label to maintain frame alignment
                        label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                        with open(label_path, "w") as _lf:
                            pass  # empty = no detection this frame
                        continue
                    img = result.orig_img.copy()
                    track_ids = result.boxes.id
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    kpts_xy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                    kpts_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None
                    if kpts_xy is None:
                        continue

                    n_inst = len(result.boxes)

                    # ── Slot assignment (replicates process_pose_file) ──
                    new_frame = [None] * max_mice
                    slot_assigned = [False] * max_mice

                    # Step 1: Match existing slot IDs
                    for inst_idx in range(n_inst):
                        tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                        if tid is None:
                            continue
                        for slot_idx, last_id in enumerate(slot_ids):
                            if last_id is not None and tid == last_id and not slot_assigned[slot_idx]:
                                new_frame[slot_idx] = inst_idx
                                slot_assigned[slot_idx] = True
                                break

                    # Step 2: Place unmatched into free slots
                    for inst_idx in range(n_inst):
                        if inst_idx in new_frame:
                            continue
                        tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                        for slot_idx in range(max_mice):
                            if not slot_assigned[slot_idx]:
                                new_frame[slot_idx] = inst_idx
                                slot_assigned[slot_idx] = True
                                if tid is not None:
                                    slot_ids[slot_idx] = tid
                                break

                    # Update slot IDs from this frame's track IDs
                    for slot_idx in range(max_mice):
                        if new_frame[slot_idx] is not None:
                            tid = int(track_ids[new_frame[slot_idx]].item()) if track_ids is not None else None
                            if tid is not None:
                                slot_ids[slot_idx] = tid

                    # ── Left/right identity: first frame with ≥2 instances ──
                    if not identity_initialized:
                        active_slots = [(si, new_frame[si]) for si in range(max_mice) if new_frame[si] is not None]
                        if len(active_slots) >= 2:
                            # Sort by center-x: left → right
                            active_slots.sort(key=lambda si: boxes_xyxy[si[1], 0] + boxes_xyxy[si[1], 2])
                            # Reorder slot IDs for future frames
                            new_slot_ids = [None] * max_mice
                            for new_slot, (old_slot, _) in enumerate(active_slots):
                                new_slot_ids[new_slot] = slot_ids[old_slot]
                            slot_ids = new_slot_ids
                            # Also reorder current frame's new_frame to match
                            reordered_frame = [None] * max_mice
                            for new_slot, (old_slot, _) in enumerate(active_slots):
                                reordered_frame[new_slot] = new_frame[old_slot]
                            new_frame = reordered_frame
                            identity_initialized = True

                    # ── Build labeled instances ──
                    instances = []
                    for slot_idx in range(max_mice):
                        inst_idx = new_frame[slot_idx]
                        if inst_idx is None:
                            continue
                        label = f"Mouse {slot_idx + 1}"
                        color = self._INSTANCE_COLORS[slot_idx % len(self._INSTANCE_COLORS)]
                        all_kps = kpts_xy[inst_idx]
                        all_conf = kpts_conf[inst_idx] if kpts_conf is not None else np.ones(len(all_kps))
                        instances.append({
                            "label": label, "color": color,
                            "keypoints": all_kps, "conf": all_conf,
                            "tail_kps": None, "tail_conf": None,
                        })

                    # ── Tail matching (only for two-model mode) ──
                    if tail_queue is not None:
                        tail_result = tail_queue.get()
                        if tail_result is None or isinstance(tail_result, BaseException):
                            tail_result = None
                        if (tail_result is not None and tail_result.boxes is not None
                                and len(tail_result.boxes) > 0 and tail_result.keypoints is not None):
                            tail_xy = tail_result.keypoints.xy.cpu().numpy()
                            tail_conf_arr = tail_result.keypoints.conf.cpu().numpy()
                            tail_used = [False] * len(tail_xy)
                            for inst in instances:
                                if len(inst["keypoints"]) < 4:
                                    continue
                                base_pt = inst["keypoints"][3]
                                best_dist = float("inf")
                                best_ti = -1
                                for ti in range(len(tail_xy)):
                                    if tail_used[ti] or tail_conf_arr[ti][0] < 0.3:
                                        continue
                                    dist = np.sqrt((base_pt[0] - tail_xy[ti][0][0])**2
                                                   + (base_pt[1] - tail_xy[ti][0][1])**2)
                                    if dist < best_dist:
                                        best_dist = dist; best_ti = ti
                                if best_ti >= 0 and best_dist < 100:
                                    inst["tail_kps"] = tail_xy[best_ti]
                                    inst["tail_conf"] = tail_conf_arr[best_ti]
                                    tail_used[best_ti] = True

                    # FPS tracking (EMA)
                    _now = _time.time()
                    _dt = _now - _last_frame_time
                    _last_frame_time = _now
                    if _dt > 0:
                        _instant_fps = 1.0 / _dt
                        _fps_ema = (_fps_alpha * _instant_fps
                                    + (1 - _fps_alpha) * _fps_ema) if _fps_ema > 0 else _instant_fps

                    # ── Render + save + label I/O ──
                    _t0_post = _time.time()
                    if save_viz:
                        self._draw_pose_frame(img, instances, frame_number=frame_idx,
                                              fps=_fps_ema, real_fps=_interval_fps)
                        frame_path = str(frames_dir / f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        src_frames.append(frame_path)
                    frame_count += 1
                    processed_frames += 1
                    if processed_frames % 10 == 0:
                        pct = 30 + int(processed_frames / total_frames * 70)
                        self.set_progress(min(pct, 99),
                            f"{processed_frames}/{total_frames}  {_interval_fps:.0f} FPS")

                    # Record structured data
                    tracking_data[str(frame_idx)] = []
                    for slot_idx, inst in enumerate(instances):
                        inst_idx = new_frame[slot_idx]
                        entry = {
                            "slot": slot_idx, "label": inst["label"],
                            "keypoints": [[float(inst["keypoints"][k][0]), float(inst["keypoints"][k][1]),
                                           float(inst["conf"][k])] for k in range(len(inst["keypoints"]))]
                        }
                        # Save box center for trajectory visualization
                        if inst_idx is not None and inst_idx < len(boxes_xyxy):
                            x1, y1, x2, y2 = boxes_xyxy[inst_idx]
                            entry["box_center"] = [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                        tracking_data[str(frame_idx)].append(entry)

                    # Save keypoint labels (YOLO format) for downstream use
                    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                    label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                    with open(label_path, "w") as lf:
                        for slot_idx in range(max_mice):
                            inst_idx = new_frame[slot_idx]
                            if inst_idx is None:
                                continue
                            tid = slot_ids[slot_idx]
                            cls_id = int(result.boxes.cls[inst_idx].item()) if result.boxes.cls is not None else 0
                            line_parts = [cls_id] + boxes_xywhn[inst_idx].tolist()
                            line_parts += kpts_xy[inst_idx].flatten().tolist()
                            line_parts += kpts_conf[inst_idx].tolist()
                            # Append matched tail keypoints if available
                            if slot_idx < len(instances) and instances[slot_idx].get("tail_kps") is not None:
                                line_parts += instances[slot_idx]["tail_kps"].flatten().tolist()
                                line_parts += instances[slot_idx]["tail_conf"].tolist()
                            line_parts.append(tid if tid is not None else slot_idx + 1)
                            lf.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in line_parts) + "\n")
                    n_label_files += 1
                    _t["post"] += _time.time() - _t0_post

                    # ── Periodic timing log (every 30 frames) ──
                    if frame_count % 30 == 0 and frame_count > 0:
                        _now = _time.time()
                        _df = frame_count - _prev_profile_frame
                        _dt = _now - _prev_profile_time
                        _interval_fps = _df / _dt if _dt > 0 else 0
                        _prev_profile_frame = frame_count
                        _prev_profile_time = _now
                        fc = max(frame_count, 1)
                        _df = max(frame_count - _prev_profile_frame, 1)
                        _dt = max(_now - _prev_profile_time, 0.001)
                        _frame_ms = _dt * 1000 / _df
                        self.log(
                            f"  [Profile #{frame_idx}] fps={_interval_fps:.1f} ({_frame_ms:.0f}ms/frame) | "
                            f"gpu={_t['model']/fc*1000:.0f}ms "
                            f"post={_t['post']/fc*1000:.1f}ms",
                            20)

                    frame_idx += 1

                # Wait for tail thread
                if tail_thread is not None:
                    tail_thread.join(timeout=10)
                    while not tail_queue.empty():
                        try: tail_queue.get_nowait()
                        except: break

                # Save tracking JSON
                tj_path = src_out / "tracking_data.json"
                with open(tj_path, "w", encoding="utf-8") as f:
                    _json.dump(tracking_data, f, ensure_ascii=False, indent=2)
                src_tracking_json = str(tj_path)

                # Save compact trajectory JSON for visualization
                img_h, img_w = img.shape[:2]
                trajectory_data = {
                    "frame_width": int(img_w), "frame_height": int(img_h),
                    "fps": 30.0,
                    "mice": {}
                }
                for frame_key in sorted(tracking_data.keys(), key=int):
                    for inst in tracking_data[frame_key]:
                        mid = str(inst["slot"])
                        if mid not in trajectory_data["mice"]:
                            trajectory_data["mice"][mid] = {
                                "label": inst["label"],
                                "color_bgr": list(InferenceWorker._INSTANCE_COLORS[
                                    inst["slot"] % len(InferenceWorker._INSTANCE_COLORS)]),
                                "trajectory": []
                            }
                        bc = inst.get("box_center")
                        if bc:
                            trajectory_data["mice"][mid]["trajectory"].append(
                                [int(frame_key), bc[0], bc[1]])
                traj_path = src_out / "trajectory.json"
                with open(traj_path, "w", encoding="utf-8") as f:
                    _json.dump(trajectory_data, f, ensure_ascii=False)
                src_trajectory_json = str(traj_path)
                total_s = _time.time() - _t_start
                fc = max(frame_count, 1)
                real_fps = fc / total_s if total_s > 0 else 0
                other_ms = (total_s * 1000 - sum(_t.values())) / fc
                self.log(
                    f"  Video done: {frame_count} frames in {total_s:.1f}s "
                    f"(real {real_fps:.1f} FPS)", 20)
                self.log(
                    f"  [Timing] per-frame: "
                    f"model={_t['model']/fc*1000:.0f}ms "
                    f"post={_t['post']/fc*1000:.1f}ms "
                    f"other={other_ms:.1f}ms",
                    20)
                source_results.append({
                    "name": src_path.name, "type": "video",
                    "frames": src_frames,
                    "tracking_json": src_tracking_json,
                    "trajectory_json": src_trajectory_json,
                    "frame_width": int(img_w), "frame_height": int(img_h),
                })
                # Show this source immediately in the GUI
                self.partial_result.emit({
                    "mode": "Pose",
                    "sources": [source_results[-1]],
                    "is_partial": True,
                })

            else:
                # ── Image folder: predict + custom rendering ──
                results = model.predict(
                    source=str(src_path), save=False, save_txt=False,
                    project=str(out_dir), name=f"source_{i}",
                    exist_ok=True, verbose=False, imgsz=640,
                )
                for img_idx, result in enumerate(results):
                    if self._cancelled:
                        self.log("  Cancelled by user", 30)
                        break
                    img = result.orig_img.copy()
                    kpts_xy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                    kpts_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None
                    if kpts_xy is None or result.boxes is None or len(result.boxes) == 0:
                        continue

                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    n_inst = len(kpts_xy)

                    # Sort instances left→right by center-x
                    inst_indices = sorted(range(n_inst), key=lambda ii: boxes_xyxy[ii, 0] + boxes_xyxy[ii, 2])
                    instances = []
                    for slot_idx, inst_idx in enumerate(inst_indices):
                        label = f"Mouse {slot_idx + 1}"
                        color = self._INSTANCE_COLORS[slot_idx % len(self._INSTANCE_COLORS)]
                        all_kps = kpts_xy[inst_idx]
                        all_conf = kpts_conf[inst_idx] if kpts_conf is not None else np.ones(len(all_kps))
                        instances.append({
                            "label": label, "color": color,
                            "keypoints": all_kps, "conf": all_conf,
                            "tail_kps": None, "tail_conf": None,
                        })

                    if save_viz:
                        self._draw_pose_frame(img, instances, frame_number=img_idx)
                        frame_path = str(frames_dir / f"frame_{img_idx:06d}.jpg")
                        cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        src_frames.append(frame_path)

                    processed_frames += 1
                    if processed_frames % 10 == 0:
                        pct = 30 + int(processed_frames / total_frames * 70)
                        self.set_progress(min(pct, 99), f"Processing... {processed_frames}/{total_frames}")

                    # Save YOLO-format label
                    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                    label_path = labels_dir / f"frame_{img_idx:06d}.txt"
                    with open(label_path, "w") as lf:
                        for slot_idx, inst_idx in enumerate(inst_indices):
                            cls_id = int(result.boxes.cls[inst_idx].item()) if result.boxes.cls is not None else 0
                            line_parts = [cls_id] + boxes_xywhn[inst_idx].tolist()
                            line_parts += kpts_xy[inst_idx].flatten().tolist()
                            line_parts += kpts_conf[inst_idx].tolist()
                            lf.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in line_parts) + "\n")
                    n_label_files += 1

                self.log(f"  Folder done: {len(src_frames)} images rendered", 20)
                source_results.append({
                    "name": src_path.name, "type": "folder",
                    "frames": src_frames,
                    "tracking_json": None,
                })
                self.partial_result.emit({
                    "mode": "Pose",
                    "sources": [source_results[-1]],
                    "is_partial": True,
                })

        # Build flat lists for backward compat
        all_frames = []
        all_tracking = []
        all_trajectory = []
        for sr in source_results:
            all_frames.extend(sr["frames"])
            if sr.get("tracking_json"):
                all_tracking.append(sr["tracking_json"])
            if sr.get("trajectory_json"):
                all_trajectory.append(sr["trajectory_json"])

        self.set_progress(100, f"Inference complete — {n_label_files} labels, {processed_frames} frames")

        self.result_ready.emit({
            "success": True, "mode": "Pose",
            "n_files": n_label_files, "output_dir": str(out_dir),
            "sources": source_results,
            "sample_images": all_frames,
            "tracking_jsons": all_tracking,
            "trajectory_jsons": all_trajectory,
            "message": f"Pose inference complete: {n_label_files} labels → {out_dir}",
        })

    # ── Live camera Pose inference ──────────────────────────────────

    @staticmethod
    def _behavior_worker_process(kp_queue, res_queue, stop_event,
                                  run_dir_str, max_instances, target_fps,
                                  log_queue, skip_temporal):
        """Run behavior inference in a SEPARATE PROCESS (no GIL contention).

        Receives keypoint windows via kp_queue, runs the full SABER pipeline,
        and sends label results back via res_queue.  All models are loaded
        independently in this process so CPU-heavy factor computation
        never blocks the main pose loop.
        """
        import time, json, logging as _logging, yaml as _yaml
        import numpy as np
        from pathlib import Path

        _logging.getLogger("inference").setLevel(_logging.WARNING)
        _logging.getLogger("src.factor_engine").setLevel(_logging.WARNING)

        run_path = Path(run_dir_str)

        # Load config
        merged_yaml = run_path / "configs" / "merged_config.yaml"
        if merged_yaml.exists():
            with open(merged_yaml, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
        else:
            built_cfg = run_path / "configs" / "built_config.json"
            if built_cfg.exists():
                with open(built_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                log_queue.put(f"[Behavior] ERROR: no config found in {run_path}/configs/")
                return

        from inference.inference import (
            InferencePipeline, determine_mouse_order, build_keypoints_for_center,
        )
        from src.data_loader import FeatureNormalizer

        pipeline = InferencePipeline(run_path, cfg)

        # Move temporal NN to CPU in this process too
        import torch as _torch
        if (hasattr(pipeline, 'temporal_nn_model')
                and pipeline.temporal_nn_model is not None):
            pipeline.temporal_nn_model = pipeline.temporal_nn_model.cpu()
            _torch.cuda.empty_cache()

        num_keypoints = 10
        center_order = list(range(max_instances))

        log_queue.put(f"[Behavior] subprocess ready ({max_instances} mice, skip_temporal={skip_temporal})")

        class _KpInst:
            __slots__ = ('xy', 'xywh', 'points_xy')
            def __init__(self, xy, xywh, points_xy):
                self.xy = xy; self.xywh = xywh; self.points_xy = points_xy

        while not stop_event.is_set():
            try:
                window_data = kp_queue.get(timeout=1.0)
            except Exception:
                continue

            if window_data is None:  # sentinel
                break

            try:
                t0 = time.time()

                # Reconstruct _KpInst objects from serialized data
                window = []
                for frame_list in window_data:
                    frame_insts = []
                    for item in frame_list:
                        if item is not None:
                            frame_insts.append(_KpInst(
                                item['xy'], item['xywh'], item['points_xy']))
                        else:
                            frame_insts.append(None)
                    window.append(frame_insts)

                # Build video_data
                video_data = []
                for frame_insts in window:
                    frame_list = [frame_insts[m] if m < len(frame_insts) else None
                                  for m in range(max_instances)]
                    video_data.append(frame_list)

                center_order = determine_mouse_order([video_data], max_instances)

                new_labels = {}
                import torch as _torch2
                _cuda_avail = _torch2.cuda.is_available
                _torch2.cuda.is_available = lambda: False
                try:
                    for logical_id in range(max_instances):
                        actual_center_id = center_order[logical_id]
                        dummy_kp = np.zeros((len(window_data), 1), dtype=np.float32)
                        kp_raw, flat_attrs = build_keypoints_for_center(
                            keypoints_full=dummy_kp, merged_data=[video_data],
                            center_id=actual_center_id, max_instance_num=max_instances,
                            num_keypoints=num_keypoints, fps=target_fps, normalizer=None)

                        if kp_raw.size == 0:
                            continue

                        normalizer = FeatureNormalizer().fit(kp_raw)
                        kp_norm = normalizer.transform(kp_raw)
                        proba = pipeline.predict(
                            kp_norm, flat_attrs, center_id=actual_center_id,
                            skip_temporal=skip_temporal)
                        labels_arr = pipeline.decode_to_labels(proba)
                        if len(labels_arr) > 0:
                            new_labels[str(logical_id)] = [int(x) for x in labels_arr]
                finally:
                    _torch2.cuda.is_available = _cuda_avail

                new_class_names = [
                    pipeline.id_to_name.get(i, f"class_{i}")
                    for i in sorted(pipeline.id_to_name)]

                elapsed = time.time() - t0
                log_queue.put(
                    f"  [Behavior] {len(window_data)}fr → "
                    f"{len(new_class_names)}cls {elapsed:.1f}s (subprocess)")

                # Send results back (non-blocking, drop old if UI hasn't consumed)
                try:
                    res_queue.put_nowait((new_labels, new_class_names))
                except Exception:
                    pass

            except Exception as _be:
                import traceback as _tb
                log_queue.put(f"  [Behavior] ERROR: {_be}\n{_tb.format_exc()}")

        log_queue.put("[Behavior] subprocess stopped")

    @staticmethod
    def _do_auto_save(session_dir, label_history, class_names, fps,
                      save_csv=True, save_raster=True, total_frames=0):
        """Save accumulated behavior labels as CSV and/or raster PNG.

        Called periodically during live E2E and at session end.
        label_history: list of (frame_idx, {mouse_id_str: label_int})
        total_frames: fixed timeline width in frames (e.g. 10min*60*30fps=18000).
                      When > 0, raster is fixed-width: left fills with data, right stays white.
        """
        import csv, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        session_dir = Path(session_dir)
        n_frames = len(label_history)
        if n_frames == 0:
            return
        # Find mouse IDs from first entry that actually has labels
        mouse_ids = []
        for _, entry in label_history:
            if entry:
                mouse_ids = sorted(entry.keys(), key=int)
                break
        if not mouse_ids:
            return
        n_mice = len(mouse_ids)
        n_classes = len(class_names)

        # ── Save CSV (always actual data only, no padding) ──
        if save_csv:
            csv_path = session_dir / "behavior.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                header = ["frame", "time_sec"]
                for mid in mouse_ids:
                    header += [f"label_mouse{int(mid)+1}", f"behavior_mouse{int(mid)+1}"]
                writer.writerow(header)
                for frame_idx, entry in label_history:
                    t_sec = round(frame_idx / fps, 4) if fps > 0 else 0
                    row = [frame_idx, t_sec]
                    for mid in mouse_ids:
                        lbl = entry.get(mid, -1)
                        name = class_names[lbl] if 0 <= lbl < n_classes else "unknown"
                        row += [lbl, name]
                    writer.writerow(row)

        # ── Save raster PNG (fixed-width timeline, fills left→right) ──
        if save_raster and n_classes > 0:
            # Use fixed total width if specified, otherwise fit to data
            if total_frames > 0:
                raster_width = total_frames
            else:
                raster_width = max(label_history[-1][0] + 1, 1)

            # Build per-mouse label arrays (padded to raster_width)
            label_arrays = {}
            for mid in mouse_ids:
                arr = np.full(raster_width, -1, dtype=int)
                for frame_idx, entry in label_history:
                    if mid in entry and frame_idx < raster_width:
                        arr[frame_idx] = entry[mid]
                label_arrays[mid] = arr

            # Plot
            cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
            strip_h = 0.5
            legend_h = 0.45
            total_h = strip_h * n_mice + legend_h
            fig, axes = plt.subplots(n_mice, 1, figsize=(8, total_h),
                                     sharex=True, squeeze=False,
                                     gridspec_kw={'hspace': 0.20, 'top': 0.96,
                                                  'bottom': legend_h / total_h,
                                                  'left': 0.0, 'right': 1.0})
            for row_idx, mid in enumerate(mouse_ids):
                ax = axes[row_idx][0]
                arr = label_arrays[mid]
                colors_arr = cmap([c / max(1, n_classes - 1) for c in range(n_classes)])
                img = np.zeros((1, len(arr), 3))
                for c in range(n_classes):
                    img[0, arr == c] = colors_arr[c][:3]
                img[0, arr < 0] = [1, 1, 1]  # unlabeled → white
                ax.imshow(img, aspect='auto', interpolation='nearest')
                ax.set_yticks([]); ax.set_xticks([]); ax.margins(0)
            patches = [mpatches.Patch(color=cmap(i), label=class_names[i])
                       for i in range(n_classes)]
            fig.legend(handles=patches, loc="lower center",
                       ncol=min(n_classes, 10), fontsize=9, frameon=False,
                       handlelength=1.2, handleheight=1.2, columnspacing=0.5,
                       markerscale=1.0, borderpad=0.1, labelspacing=0.2)
            raster_path = str(session_dir / "live_raster.png")
            fig.savefig(raster_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def _run_live_pose(self):
        """Live camera pose inference — real-time YOLO tracking from webcam.

        Opens a camera device, runs YOLO pose tracking on each frame,
        draws visualizations, encodes to JPEG bytes in-memory (zero disk I/O),
        and emits via partial_result for live UI display.
        """
        import cv2, time, json as _json

        pose_run_dir = self._params.get("pose_run_dir", "")
        if not pose_run_dir:
            self.error.emit("Pose Run Dir not specified")
            return

        run_path = Path(pose_run_dir)
        weights = run_path / "weights" / "best.pt"
        if not weights.exists():
            alt = list(run_path.glob("*.pt"))
            weights = alt[0] if alt else weights
        if not weights.exists():
            self.error.emit(f"Pose weights not found: {weights}")
            return

        camera_id = int(self._params.get("camera_id", 0))
        cam_width = int(self._params.get("camera_width", 640))
        cam_height = int(self._params.get("camera_height", 480))
        target_fps = int(self._params.get("camera_fps", 30))

        self.set_progress(5, "Opening camera...")
        self.log(f"Live Pose: camera={camera_id} {cam_width}×{cam_height} @{target_fps}fps", 20)

        # Open camera (use DSHOW backend on Windows for better compatibility)
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {camera_id}")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.log(f"Camera opened: {actual_w}×{actual_h} @{actual_fps:.0f}fps", 20)

        self.set_progress(10, "Loading YOLO model...")
        from src.pose_trainer import _ensure_tmp_registered
        _ensure_tmp_registered()
        from ultralytics import YOLO
        model = YOLO(str(weights))

        # Optional tail model
        tail_model = None
        if bool(self._params.get("tail_run_dir", "") and str(self._params.get("tail_run_dir", "")).strip()):
            tail_run_dir = self._params.get("tail_run_dir", "")
            if tail_run_dir and str(tail_run_dir).strip():
                tail_weights = Path(tail_run_dir) / "weights" / "best.pt"
                if tail_weights.exists():
                    tail_model = YOLO(str(tail_weights))
                    self.log("Two-model live mode: body + tail", 20)

        # Tracking state (same slot-assignment logic as batch mode)
        max_mice = int(self._params.get("max_mice", 2))
        slot_ids = [None] * max_mice
        identity_initialized = False

        frame_count = 0
        max_buffer = 300  # sliding window for trajectory

        _last_frame_time = time.time()
        _fps_ema = 0.0
        _fps_alpha = 0.1
        _interval_fps = 0.0
        _prev_profile_time = time.time()

        self.set_progress(20, "Live capture running...")
        self.log("Live capture started — zero disk I/O — click Stop to end", 20)

        # Pre-allocate trajectory dict (incremental, not rebuilt per frame)
        trajectory_data = {
            "frame_width": actual_w, "frame_height": actual_h,
            "fps": float(target_fps), "mice": {},
        }
        for slot_idx in range(max_mice):
            mid = str(slot_idx)
            trajectory_data["mice"][mid] = {
                "label": f"Mouse {slot_idx + 1}",
                "color_bgr": list(self._INSTANCE_COLORS[
                    slot_idx % len(self._INSTANCE_COLORS)]),
                "trajectory": [],
            }

        try:
            while not self._cancelled:
                ret, frame = cap.read()
                if not ret:
                    self.log("Camera read failed — stream ended", 30)
                    break

                # YOLO tracking (single-frame call with persist=True, FP16)
                results = model.track(
                    frame, persist=True, verbose=False,
                    imgsz=640, half=True,
                )
                result = results[0] if isinstance(results, list) else results

                img = frame.copy()
                instances = []
                boxes_xyxy = np.zeros((0, 4))

                if result.boxes is not None and len(result.boxes) > 0:
                    track_ids = result.boxes.id
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    kpts_xy = (result.keypoints.xy.cpu().numpy()
                               if result.keypoints is not None else None)
                    kpts_conf = (result.keypoints.conf.cpu().numpy()
                                 if result.keypoints is not None else None)

                    if kpts_xy is not None:
                        n_inst = len(result.boxes)

                        # ── Slot assignment ──
                        new_frame = [None] * max_mice
                        slot_assigned = [False] * max_mice

                        for inst_idx in range(n_inst):
                            tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                            if tid is None:
                                continue
                            for slot_idx, last_id in enumerate(slot_ids):
                                if (last_id is not None and tid == last_id
                                        and not slot_assigned[slot_idx]):
                                    new_frame[slot_idx] = inst_idx
                                    slot_assigned[slot_idx] = True
                                    break

                        for inst_idx in range(n_inst):
                            if inst_idx in new_frame:
                                continue
                            tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                            for slot_idx in range(max_mice):
                                if not slot_assigned[slot_idx]:
                                    new_frame[slot_idx] = inst_idx
                                    slot_assigned[slot_idx] = True
                                    if tid is not None:
                                        slot_ids[slot_idx] = tid
                                    break

                        for slot_idx in range(max_mice):
                            if new_frame[slot_idx] is not None:
                                tid = (int(track_ids[new_frame[slot_idx]].item())
                                       if track_ids is not None else None)
                                if tid is not None:
                                    slot_ids[slot_idx] = tid

                        # Left/right identity on first frame with ≥2 instances
                        if not identity_initialized:
                            active = [(si, new_frame[si]) for si in range(max_mice)
                                      if new_frame[si] is not None]
                            if len(active) >= 2:
                                active.sort(key=lambda si: (
                                    boxes_xyxy[si[1], 0] + boxes_xyxy[si[1], 2]))
                                new_slot_ids = [None] * max_mice
                                for ns, (os, _) in enumerate(active):
                                    new_slot_ids[ns] = slot_ids[os]
                                slot_ids = new_slot_ids
                                reordered = [None] * max_mice
                                for ns, (os, _) in enumerate(active):
                                    reordered[ns] = new_frame[os]
                                new_frame = reordered
                                identity_initialized = True

                        # Build labeled instances
                        for slot_idx in range(max_mice):
                            inst_idx = new_frame[slot_idx]
                            if inst_idx is None:
                                continue
                            label = f"Mouse {slot_idx + 1}"
                            color = self._INSTANCE_COLORS[
                                slot_idx % len(self._INSTANCE_COLORS)]
                            all_kps = kpts_xy[inst_idx]
                            all_conf = (kpts_conf[inst_idx] if kpts_conf is not None
                                        else np.ones(len(all_kps)))
                            instances.append({
                                "label": label, "color": color,
                                "keypoints": all_kps, "conf": all_conf,
                                "tail_kps": None, "tail_conf": None,
                            })

                # ── FPS tracking (EMA) ──
                _now = time.time()
                _dt = _now - _last_frame_time
                _last_frame_time = _now
                if _dt > 0:
                    _instant_fps = 1.0 / _dt
                    _fps_ema = ((_fps_alpha * _instant_fps
                                 + (1 - _fps_alpha) * _fps_ema)
                                if _fps_ema > 0 else _instant_fps)

                # ── Draw keypoints on frame ──
                self._draw_pose_frame(img, instances, frame_number=frame_count,
                                      fps=_fps_ema, real_fps=_interval_fps)

                # ── Encode to JPEG bytes IN MEMORY (zero disk I/O) ──
                _, jpeg_buf = cv2.imencode('.jpg', img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 75])
                jpeg_bytes = jpeg_buf.tobytes()

                # ── Incremental trajectory update (for viz modes: trail/heatmap/ROI) ──
                for slot_idx, inst in enumerate(instances):
                    mid = str(slot_idx)
                    if (slot_idx < len(new_frame) and new_frame[slot_idx] is not None
                            and len(boxes_xyxy) > 0
                            and new_frame[slot_idx] < len(boxes_xyxy)):
                        x1, y1, x2, y2 = boxes_xyxy[new_frame[slot_idx]]
                        trajectory_data["mice"][mid]["trajectory"].append(
                            [frame_count, float((x1 + x2) / 2), float((y1 + y2) / 2)])

                frame_count += 1

                # Periodic FPS log
                if frame_count % 30 == 0:
                    _now2 = time.time()
                    _dt2 = _now2 - _prev_profile_time
                    _interval_fps = 30 / _dt2 if _dt2 > 0 else 0
                    _prev_profile_time = _now2
                    self.log(
                        f"  Live Pose #{frame_count}  fps={_fps_ema:.1f}  "
                        f"throughput={_interval_fps:.1f}", 20)

                # ── Emit throttled: every 2 frames (~15 Hz display) ──
                # Only send the LATEST frame JPEG + trajectory, no file paths
                if frame_count % 2 != 0:
                    continue

                self.partial_result.emit({
                    "mode": "Pose",
                    "is_partial": True,
                    "is_live": True,
                    "frame_jpeg": jpeg_bytes,
                    "frame_idx": frame_count - 1,
                    "fps": round(_fps_ema, 1),
                    "trajectory_data": trajectory_data,
                    "frame_width": actual_w,
                    "frame_height": actual_h,
                })

        finally:
            cap.release()

        self.set_progress(100, f"Live Pose ended — {frame_count} frames")
        self.result_ready.emit({
            "success": True, "mode": "Pose",
            "n_files": frame_count,
            "message": f"Live Pose: {frame_count} frames captured",
        })

    # ── Live camera End-to-End inference ─────────────────────────────

    def _run_live_end_to_end(self):
        """Live End-to-End: camera → YOLO pose + behavior in SEPARATE PROCESS.

        ZERO file I/O — frames are encoded to JPEG bytes in memory.
        Pose estimation runs uninterrupted; behavior inference runs in a
        multiprocessing.Process to avoid GIL/CPU contention with YOLO.
        """
        import cv2, time, json as _json, multiprocessing as _mp

        pose_run_dir = self._params.get("pose_run_dir", "")
        run_dir = self._params.get("run_dir", "")
        if not pose_run_dir:
            self.error.emit("Pose Run Dir not specified")
            return
        if not run_dir:
            self.error.emit("Behavior Run Dir not specified")
            return

        run_path = Path(run_dir)
        if not run_path.exists():
            self.error.emit(f"Run directory not found: {run_dir}")
            return

        pose_weights = Path(pose_run_dir) / "weights" / "best.pt"
        if not pose_weights.exists():
            alt = list(Path(pose_run_dir).glob("*.pt"))
            pose_weights = alt[0] if alt else pose_weights
        if not pose_weights.exists():
            self.error.emit(f"Pose weights not found: {pose_weights}")
            return

        camera_id = int(self._params.get("camera_id", 0))
        cam_width = int(self._params.get("camera_width", 640))
        cam_height = int(self._params.get("camera_height", 480))
        target_fps = int(self._params.get("camera_fps", 30))

        self.set_progress(5, "Opening camera...")
        self.log(f"Live E2E: camera={camera_id} {cam_width}×{cam_height}", 20)

        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {camera_id}")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.log(f"Camera opened: {actual_w}×{actual_h} @{actual_fps:.0f}fps", 20)

        # Load YOLO
        self.set_progress(10, "Loading YOLO model...")
        from src.pose_trainer import _ensure_tmp_registered
        _ensure_tmp_registered()
        from ultralytics import YOLO
        model = YOLO(str(pose_weights))

        # Load SABER pipeline config (for metadata only — subprocess loads its own)
        self.set_progress(15, "Loading SABER pipeline...")
        import yaml as _yaml
        merged_yaml = run_path / "configs" / "merged_config.yaml"
        if merged_yaml.exists():
            with open(merged_yaml, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
        else:
            built_cfg = run_path / "configs" / "built_config.json"
            if built_cfg.exists():
                with open(built_cfg, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
            else:
                self.error.emit("No config found in run/configs/")
                return

        from inference.inference import InferencePipeline
        pipeline = InferencePipeline(run_path, cfg)
        max_instances = cfg.get("dataset_config", [{}])[0].get("max_instances_num", 2)

        # Move temporal NN to CPU in main process too
        import torch as _torch
        if (hasattr(pipeline, 'temporal_nn_model')
                and pipeline.temporal_nn_model is not None):
            pipeline.temporal_nn_model = pipeline.temporal_nn_model.cpu()
            _torch.cuda.empty_cache()
            self.log("Moved temporal NN to CPU → freed GPU memory for YOLO", 20)
        self.log(f"SABER pipeline loaded ({max_instances} mice)", 20)

        # ── Minimal keypoint-instance holder ──
        class _KpInst:
            __slots__ = ('xy', 'xywh', 'points_xy')
            def __init__(self, xy, xywh, points_xy):
                self.xy = xy; self.xywh = xywh; self.points_xy = points_xy

        # Tracking state
        max_mice = int(self._params.get("max_mice", 2))
        slot_ids = [None] * max_mice
        identity_initialized = False

        frame_count = 0
        max_buffer = 300

        # ── In-memory keypoint buffer ──
        kp_buffer = []
        import threading as _th
        buf_lock = _th.Lock()

        # ── Multiprocessing: behavior in SEPARATE PROCESS (no GIL contention!) ──
        ctx = _mp.get_context('spawn')
        kp_queue = ctx.Queue(maxsize=2)     # keypoints IN
        res_queue = ctx.Queue(maxsize=2)    # labels OUT
        log_queue = ctx.Queue(maxsize=50)   # log messages from subprocess
        beh_stop = ctx.Event()
        beh_shared = {"labels": {}, "class_names": []}
        beh_lock = _th.Lock()
        behavior_window = 90
        skip_temporal = self._params.get("skip_temporal", True)

        beh_process = ctx.Process(
            target=InferenceWorker._behavior_worker_process,
            args=(kp_queue, res_queue, beh_stop,
                  str(run_path), max_instances, target_fps,
                  log_queue, skip_temporal),
            daemon=True,
        )
        beh_process.start()
        self.log("Behavior process spawned (multiprocessing, zero GIL contention)", 20)

        # ── Per-step timing ──
        _last_frame_time = time.time()
        _fps_ema = 0.0; _fps_alpha = 0.1
        _interval_fps = 0.0
        _prev_profile_time = time.time()
        _t = {"read": 0.0, "track": 0.0, "draw": 0.0, "enc": 0.0,
              "traj": 0.0, "emit": 0.0}
        _last_beh_push = time.time()
        beh_interval_s = 2.0  # push keypoints to subprocess every 2s

        # ── Auto-save + live raster state ──
        auto_save_interval_s = int(self._params.get("auto_save_interval_min", 10)) * 60
        auto_save_enabled = auto_save_interval_s > 0
        save_behavior_csv = self._params.get("auto_save_behavior_csv", True)
        save_raster_png = self._params.get("auto_save_raster_png", True)
        auto_save_dir = str(self._params.get("auto_save_dir", "live_output") or "live_output")
        if auto_save_enabled:
            save_path = Path(auto_save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            import datetime as _dt
            session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = save_path / f"session_{session_ts}"
            session_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Auto-save enabled: every {auto_save_interval_s//60}min -> {session_dir}", 20)
        else:
            session_dir = None
        _last_autosave_time = time.time()
        _session_start_time = time.time()  # track when session started
        # Label history: list of (frame_idx, {mouse_id_str: label_int})
        label_history = []
        class_names_for_raster = []

        self.set_progress(20, "Live E2E running (behavior in subprocess, no GIL)...")
        self.log("Live E2E capture started — zero disk I/O — click Stop to end", 20)

        # Pre-allocate trajectory dict (incremental updates only)
        trajectory_data = {
            "frame_width": actual_w, "frame_height": actual_h,
            "fps": float(target_fps), "mice": {},
        }
        for slot_idx in range(max_mice):
            mid = str(slot_idx)
            trajectory_data["mice"][mid] = {
                "label": f"Mouse {slot_idx + 1}",
                "color_bgr": list(self._INSTANCE_COLORS[
                    slot_idx % len(self._INSTANCE_COLORS)]),
                "trajectory": [],
            }

        try:
            while not self._cancelled:
                _t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    self.log("Camera read failed", 30)
                    break
                _t["read"] += time.time() - _t0

                _t0 = time.time()
                results = model.track(frame, persist=True, verbose=False,
                                      imgsz=640, half=True)
                _t["track"] += time.time() - _t0
                result = results[0] if isinstance(results, list) else results

                _t0 = time.time()
                img = frame.copy()
                instances = []
                boxes_xyxy = np.zeros((0, 4))
                boxes_xywhn = np.zeros((0, 4))
                kpts_xy = None
                kpts_conf = None
                new_frame = [None] * max_mice

                if result.boxes is not None and len(result.boxes) > 0:
                    track_ids = result.boxes.id
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                    kpts_xy = (result.keypoints.xy.cpu().numpy()
                               if result.keypoints is not None else None)
                    kpts_conf = (result.keypoints.conf.cpu().numpy()
                                 if result.keypoints is not None else None)

                    if kpts_xy is not None:
                        n_inst = len(result.boxes)
                        slot_assigned = [False] * max_mice

                        for inst_idx in range(n_inst):
                            tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                            if tid is None:
                                continue
                            for slot_idx, last_id in enumerate(slot_ids):
                                if (last_id is not None and tid == last_id
                                        and not slot_assigned[slot_idx]):
                                    new_frame[slot_idx] = inst_idx
                                    slot_assigned[slot_idx] = True
                                    break

                        for inst_idx in range(n_inst):
                            if inst_idx in new_frame:
                                continue
                            tid = int(track_ids[inst_idx].item()) if track_ids is not None else None
                            for slot_idx in range(max_mice):
                                if not slot_assigned[slot_idx]:
                                    new_frame[slot_idx] = inst_idx
                                    slot_assigned[slot_idx] = True
                                    if tid is not None:
                                        slot_ids[slot_idx] = tid
                                    break

                        for slot_idx in range(max_mice):
                            if new_frame[slot_idx] is not None:
                                tid = (int(track_ids[new_frame[slot_idx]].item())
                                       if track_ids is not None else None)
                                if tid is not None:
                                    slot_ids[slot_idx] = tid

                        if not identity_initialized:
                            active = [(si, new_frame[si]) for si in range(max_mice)
                                      if new_frame[si] is not None]
                            if len(active) >= 2:
                                active.sort(key=lambda si: (
                                    boxes_xyxy[si[1], 0] + boxes_xyxy[si[1], 2]))
                                new_slot_ids = [None] * max_mice
                                for ns, (os, _) in enumerate(active):
                                    new_slot_ids[ns] = slot_ids[os]
                                slot_ids = new_slot_ids
                                reordered = [None] * max_mice
                                for ns, (os, _) in enumerate(active):
                                    reordered[ns] = new_frame[os]
                                new_frame = reordered
                                identity_initialized = True

                        for slot_idx in range(max_mice):
                            inst_idx = new_frame[slot_idx]
                            if inst_idx is None:
                                continue
                            label = f"Mouse {slot_idx + 1}"
                            color = self._INSTANCE_COLORS[
                                slot_idx % len(self._INSTANCE_COLORS)]
                            all_kps = kpts_xy[inst_idx]
                            all_conf = (kpts_conf[inst_idx] if kpts_conf is not None
                                        else np.ones(len(all_kps)))
                            instances.append({
                                "label": label, "color": color,
                                "keypoints": all_kps, "conf": all_conf,
                                "tail_kps": None, "tail_conf": None,
                            })

                # ── Accumulate keypoints IN MEMORY ──
                frame_kp_insts = []
                for slot_idx in range(max_mice):
                    inst_idx = new_frame[slot_idx]
                    if inst_idx is not None and kpts_xy is not None:
                        kps = kpts_xy[inst_idx]
                        bx, by, bw, bh = boxes_xywhn[inst_idx]
                        frame_kp_insts.append(_KpInst(
                            [float(bx), float(by)],
                            [float(bx), float(by), float(bw), float(bh)],
                            [[float(kps[k][0]), float(kps[k][1])] for k in range(len(kps))]))
                    else:
                        frame_kp_insts.append(None)

                with buf_lock:
                    kp_buffer.append(frame_kp_insts)
                    while len(kp_buffer) > max_buffer:
                        kp_buffer.pop(0)

                # ── Incremental trajectory update ──
                _t0 = time.time()
                for slot_idx in range(max_mice):
                    mid = str(slot_idx)
                    if (slot_idx < len(new_frame) and new_frame[slot_idx] is not None
                            and len(boxes_xyxy) > 0
                            and new_frame[slot_idx] < len(boxes_xyxy)):
                        x1, y1, x2, y2 = boxes_xyxy[new_frame[slot_idx]]
                        trajectory_data["mice"][mid]["trajectory"].append(
                            [frame_count, float((x1 + x2) / 2), float((y1 + y2) / 2)])
                # No trimming — preserve full history so undetected mice keep their data
                _t["traj"] += time.time() - _t0

                # FPS
                _now = time.time()
                _dt = _now - _last_frame_time
                _last_frame_time = _now
                if _dt > 0:
                    _instant_fps = 1.0 / _dt
                    _fps_ema = ((_fps_alpha * _instant_fps
                                 + (1 - _fps_alpha) * _fps_ema)
                                if _fps_ema > 0 else _instant_fps)

                # Draw pose overlay
                _t0 = time.time()
                self._draw_pose_frame(img, instances, frame_number=frame_count,
                                      fps=_fps_ema, real_fps=_interval_fps)
                _t["draw"] += time.time() - _t0

                # ── Encode to JPEG bytes IN MEMORY (zero disk I/O!) ──
                _t0 = time.time()
                _, jpeg_buf = cv2.imencode('.jpg', img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 75])
                jpeg_bytes = jpeg_buf.tobytes()
                _t["enc"] += time.time() - _t0

                frame_count += 1

                # ── Push keypoints to behavior subprocess (every 2s) ──
                _now = time.time()
                if _now - _last_beh_push >= beh_interval_s:
                    _last_beh_push = _now
                    with buf_lock:
                        if len(kp_buffer) >= behavior_window:
                            # Serialize for subprocess: _KpInst → dict (picklable)
                            win = []
                            for fi in kp_buffer[-behavior_window:]:
                                frame_ser = []
                                for inst in fi:
                                    if inst is not None:
                                        frame_ser.append({
                                            'xy': inst.xy, 'xywh': inst.xywh,
                                            'points_xy': inst.points_xy})
                                    else:
                                        frame_ser.append(None)
                                win.append(frame_ser)
                            try:
                                kp_queue.put_nowait(win)
                            except Exception:
                                pass  # subprocess busy, skip this cycle

                # ── Drain behavior results + subprocess logs (non-blocking) ──
                _beh_just_arrived = False
                try:
                    labels, cnames = res_queue.get_nowait()
                    with beh_lock:
                        beh_shared["labels"] = labels
                        beh_shared["class_names"] = cnames
                    # Accumulate labels for auto-save & live raster
                    if labels and auto_save_enabled:
                        class_names_for_raster = cnames
                        start_frame = max(0, frame_count - behavior_window)
                        n_label_frames = max(len(v) for v in labels.values()) if labels else 0
                        for rel_idx in range(n_label_frames):
                            abs_frame = start_frame + rel_idx
                            entry = {}
                            for mid_str, lbl_list in labels.items():
                                if rel_idx < len(lbl_list):
                                    entry[mid_str] = int(lbl_list[rel_idx])
                            if not label_history or label_history[-1][0] < abs_frame:
                                label_history.append((abs_frame, entry))
                        _beh_just_arrived = True
                except Exception:
                    pass
                # Drain log messages from subprocess
                try:
                    while True:
                        msg = log_queue.get_nowait()
                        self.log(msg, 20)
                except Exception:
                    pass

                # ── Regenerate raster on every behavior result; CSV on timer ──
                if auto_save_enabled and session_dir is not None and label_history:
                    _now = time.time()
                    # Raster: update every time new labels arrive (live feedback)
                    if _beh_just_arrived:
                        try:
                            total_frames = int(target_fps * auto_save_interval_s)
                            self._do_auto_save(
                                session_dir, label_history, class_names_for_raster,
                                target_fps, save_csv=False, save_raster=True,
                                total_frames=total_frames)
                            raster_png = str(session_dir / "live_raster.png")
                            if Path(raster_png).exists():
                                self.partial_result.emit({
                                    "mode": "End-to-End", "is_partial": True,
                                    "is_live": True, "is_behavior_update": True,
                                    "e2e_labels": beh_shared.get("labels", {}),
                                    "e2e_class_names": beh_shared.get("class_names", []),
                                    "raster_plot": raster_png,
                                    "total_frames": total_frames,
                                })
                        except Exception as _se:
                            self.log(f"Raster update failed: {_se}", 30)
                    # CSV + raster full save: on timer
                    if _now - _last_autosave_time >= auto_save_interval_s:
                        _last_autosave_time = _now
                        try:
                            total_frames = int(target_fps * auto_save_interval_s)
                            self._do_auto_save(
                                session_dir, label_history, class_names_for_raster,
                                target_fps, save_behavior_csv, save_raster_png,
                                total_frames=total_frames)
                            self.log(f"Auto-save complete ({len(label_history)} labels)", 20)
                        except Exception as _se:
                            self.log(f"Auto-save failed: {_se}", 30)

                # Periodic FPS log with per-step timing
                if frame_count % 30 == 0:
                    _now2 = time.time()
                    _dt2 = _now2 - _prev_profile_time
                    _interval_fps = 30 / _dt2 if _dt2 > 0 else 0
                    _prev_profile_time = _now2
                    n = 30
                    self.log(
                        f"  Live E2E #{frame_count}  fps={_fps_ema:.1f}  "
                        f"read={_t['read']/n*1000:.0f}ms "
                        f"track={_t['track']/n*1000:.0f}ms "
                        f"draw={_t['draw']/n*1000:.0f}ms "
                        f"enc={_t['enc']/n*1000:.0f}ms "
                        f"traj={_t['traj']/n*1000:.1f}ms "
                        f"emit={_t['emit']/n*1000:.1f}ms",
                        20)
                    for k in _t:
                        _t[k] = 0.0

                # ── Emit throttled: every 2 frames (~15 Hz display) ──
                if frame_count % 2 != 0:
                    continue

                # Read latest behavior results
                with beh_lock:
                    current_labels = dict(beh_shared.get("labels", {}))
                    current_class_names = list(beh_shared.get("class_names", []))

                _t0 = time.time()
                self.partial_result.emit({
                    "mode": "End-to-End",
                    "is_partial": True,
                    "is_live": True,
                    "frame_jpeg": jpeg_bytes,
                    "frame_idx": frame_count - 1,
                    "fps": round(_fps_ema, 1),
                    "trajectory_data": trajectory_data,
                    "frame_width": actual_w,
                    "frame_height": actual_h,
                    "e2e_labels": current_labels,
                    "e2e_class_names": current_class_names,
                })
                _t["emit"] += time.time() - _t0

        finally:
            beh_stop.set()
            try:
                kp_queue.put_nowait(None)  # sentinel to stop subprocess
            except Exception:
                pass
            cap.release()
            beh_process.join(timeout=5)
            if beh_process.is_alive():
                beh_process.terminate()
                beh_process.join(timeout=2)
            # Drain remaining logs
            try:
                while True:
                    msg = log_queue.get_nowait()
                    self.log(msg, 20)
            except Exception:
                pass

        with beh_lock:
            final_labels = dict(beh_shared.get("labels", {}))
            final_class_names = list(beh_shared.get("class_names", []))

        # Final auto-save: only save if session lasted >= the configured interval
        if auto_save_enabled and session_dir is not None and label_history:
            elapsed = time.time() - _session_start_time
            if elapsed >= auto_save_interval_s:
                try:
                    total_frames = int(target_fps * auto_save_interval_s)
                    self._do_auto_save(
                        session_dir, label_history, class_names_for_raster,
                        target_fps, save_behavior_csv, save_raster_png,
                        total_frames=total_frames)
                    raster_png = str(session_dir / "live_raster.png")
                    self.log(
                        f"Final auto-save complete -> {session_dir} "
                        f"(session: {elapsed:.0f}s, {len(label_history)} labels)", 20)
                    if Path(raster_png).exists():
                        self.partial_result.emit({
                            "mode": "End-to-End", "is_partial": True, "is_live": True,
                            "is_behavior_update": True,
                            "e2e_labels": final_labels,
                            "e2e_class_names": final_class_names,
                            "raster_plot": raster_png,
                            "total_frames": total_frames,
                        })
                except Exception as _se:
                    self.log(f"Final auto-save failed: {_se}", 30)
            else:
                # Session too short — discard unsaved data
                import shutil as _shutil
                try:
                    _shutil.rmtree(str(session_dir), ignore_errors=True)
                    self.log(
                        f"Session too short ({elapsed:.0f}s < {auto_save_interval_s}s), "
                        f"discarded -> {session_dir}", 20)
                except Exception:
                    pass

        self.set_progress(100, f"Live E2E ended — {frame_count} frames")
        self.result_ready.emit({
            "success": True, "mode": "End-to-End",
            "n_frames": frame_count,
            "e2e_labels": final_labels,
            "e2e_class_names": final_class_names,
            "message": f"Live E2E: {frame_count} frames captured",
        })

    # ── End-to-End inference (batch) ─────────────────────────────────

    def _run_e2e_demo(self):
        """Demo mode: generate sample raster + pose frames for layout tuning."""
        import tempfile, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self.set_progress(10, "Demo: generating sample data...")
        n_frames = 500
        n_classes = 7
        class_names = ["explore", "climb", "groom", "stand", "blank", "sniff", "approach"]
        cmap = plt.get_cmap("tab20", n_classes)

        # Generate fake labels (switching every ~50 frames)
        rng = np.random.default_rng(42)
        m1_labels = np.floor(np.arange(n_frames) / 60).astype(int) % n_classes
        m2_labels = (m1_labels + 3) % n_classes
        # Add some randomness
        m1_labels = np.clip(m1_labels + rng.integers(-1, 2, n_frames), 0, n_classes - 1)
        m2_labels = np.clip(m2_labels + rng.integers(-1, 2, n_frames), 0, n_classes - 1)

        # Generate raster
        tmpdir = tempfile.mkdtemp(prefix="e2e_demo_")
        raster_path = str(Path(tmpdir) / "demo_raster.png")
        n_per = n_frames
        strip_h = 0.5
        legend_h = 0.45
        total_h = strip_h * 2 + legend_h
        fig, axes = plt.subplots(2, 1, figsize=(8, total_h), sharex=True, squeeze=False,
                                 gridspec_kw={'hspace': 0.20, 'top': 0.96,
                                              'bottom': legend_h/total_h,
                                              'left': 0.0, 'right': 1.0})
        y_pred = np.concatenate([m1_labels, m2_labels])
        for mi in range(2):
            ax = axes[mi][0]
            yp = y_pred[mi * n_per:(mi + 1) * n_per]
            colors_arr = cmap([c / max(1, n_classes - 1) for c in range(n_classes)])
            img = np.zeros((1, len(yp), 3))
            for c in range(n_classes):
                img[0, yp == c] = colors_arr[c][:3]
            ax.imshow(img, aspect='auto', interpolation='nearest')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.margins(0)
        patches = [mpatches.Patch(color=cmap(i), label=class_names[i]) for i in range(n_classes)]
        fig.legend(handles=patches, loc="lower center", ncol=n_classes,
                   fontsize=9, frameon=False, handlelength=1.2, handleheight=1.2,
                   columnspacing=0.5, markerscale=1.0, borderpad=0.1, labelspacing=0.2)
        fig.savefig(raster_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Generate demo pose frames (colored rectangles with frame number)
        import cv2
        demo_frames = []
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        for i in range(n_frames):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0] = 40  # dark bg
            img[:, :, 1] = 40
            img[:, :, 2] = 40
            cv2.putText(img, f"Demo Frame {i}", (180, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2)
            fp = str(frames_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(fp, img)
            demo_frames.append(fp)

        # Emit results
        self.result_ready.emit({
            "success": True, "mode": "Pose",
            "n_files": 0, "output_dir": tmpdir,
            "sources": [{"name": "demo", "type": "video", "frames": demo_frames, "tracking_json": None}],
            "sample_images": demo_frames,
            "tracking_jsons": [],
            "message": "Demo data generated",
        })

        # Also emit behavior result with demo labels
        self.result_ready.emit({
            "success": True, "mode": "Behavior Prediction",
            "n_frames": n_frames * 2,
            "raster_plots": [raster_path],
            "output_path": str(Path(tmpdir) / "demo.csv"),
            "e2e_labels": {"0": m1_labels.tolist(), "1": m2_labels.tolist()},
            "e2e_class_names": class_names,
            "message": "Demo complete",
        })

        self.set_progress(100, "Demo ready")
        self.log("Demo mode: raster + frames generated for layout tuning", 20)

    def _split_keypoints_for_e2e(self, labels_dir: Path, src_out_dir: Path) -> tuple:
        """Split combined body+tail label files into mouse_labels/ and tail_labels/."""
        has_tail = bool(self._params.get("tail_run_dir", "") and str(self._params.get("tail_run_dir", "")).strip())
        if not has_tail:
            return str(labels_dir), str(labels_dir)

        mouse_dir = src_out_dir / "mouse_labels"
        tail_dir = src_out_dir / "tail_labels"
        mouse_dir.mkdir(parents=True, exist_ok=True)
        tail_dir.mkdir(parents=True, exist_ok=True)

        n_body, n_tail, n_per_kp = 6, 3, 3
        for lbl_file in sorted(labels_dir.glob("*.txt")):
            with open(lbl_file, "r") as f:
                lines = f.readlines()
            mouse_lines, tail_lines = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6: continue
                header = parts[:5]; kp_start = 5
                total_kp_vals = len(parts) - kp_start
                has_tid = (total_kp_vals % 3 != 0)
                end_idx = -1 if has_tid else len(parts)
                kp_vals = parts[kp_start:end_idx]
                tid = parts[-1] if has_tid else None
                bc, tc = n_body * n_per_kp, n_tail * n_per_kp
                mk = kp_vals[:bc]; tk = kp_vals[bc:bc + tc]
                if len(mk) >= bc:
                    ml = header + mk;
                    if tid: ml.append(tid)
                    mouse_lines.append(" ".join(ml) + "\n")
                if len(tk) >= tc:
                    tl = header + tk
                    if tid: tl.append(tid)
                    tail_lines.append(" ".join(tl) + "\n")
            # Always write a file per frame (even empty) to preserve frame count
            with open(mouse_dir / lbl_file.name, "w") as f:
                if mouse_lines: f.writelines(mouse_lines)
            with open(tail_dir / lbl_file.name, "w") as f:
                if tail_lines: f.writelines(tail_lines)
        self.log("  Keypoints split: body→mouse_labels, tail→tail_labels", 20)
        return str(mouse_dir), str(tail_dir)

    @staticmethod
    def _get_next_e2e_dir(base_dir: str = "runs") -> Path:
        """Auto-increment expN directories under runs/end2end/ (YOLO-style)."""
        e2e_dir = Path(base_dir) / "end2end"
        e2e_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted([
            int(d.name[3:]) for d in e2e_dir.iterdir()
            if d.is_dir() and d.name.startswith("exp") and d.name[3:].isdigit()
        ])
        next_id = existing[-1] + 1 if existing else 1
        run_dir = e2e_dir / f"exp{next_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _run_end_to_end(self):
        """End-to-End: process each source independently (pose→behavior per video)."""
        self.log("=" * 50, 20)
        self.log("End-to-End: Video → Pose → Behavior → Raster", 20)
        self.log("=" * 50, 20)

        sources = self._params.get("pose_inference_sources", [])
        sources = [s for s in sources if s and str(s).strip()]

        # Demo mode: if source is "demo", generate fake data for layout tuning
        if sources and str(sources[0]).lower() == "demo":
            self._run_e2e_demo()
            return

        if not sources:
            self.error.emit("No video sources specified")
            return

        # ── Create per-run output directory: runs/end2end/expN ──
        e2e_run_dir = self._get_next_e2e_dir("runs")
        # Subdirectories
        pose_dir    = e2e_run_dir / "pose"
        behavior_dir = e2e_run_dir / "behavior"
        pose_dir.mkdir(exist_ok=True)
        behavior_dir.mkdir(exist_ok=True)
        self.log(f"Output directory: {e2e_run_dir}", 20)

        orig_sources = list(sources)

        for src_idx, src in enumerate(sources):
            src_name = Path(src).stem
            self.log(f"\n--- [{src_idx+1}/{len(sources)}] {Path(src).name} ---", 20)
            pct_base = int(src_idx / len(sources) * 100)

            # Step 1: Pose inference → runs/end2end/expN/pose/src_{idx}/
            # _run_pose_inference creates source_0/{labels,frames}/ under viz_output_dir
            self.set_progress(pct_base, f"[{src_idx+1}/{len(sources)}] Pose tracking...")
            self._params["pose_inference_sources"] = [src]
            self._params["viz_output_dir"] = str(pose_dir / f"src_{src_idx}")
            self._run_pose_inference()

            # Step 2: Split keypoints
            src_out = pose_dir / f"src_{src_idx}" / "source_0"
            labels_dir = src_out / "labels"
            if not labels_dir.is_dir():
                self.log("  No labels found, skipping", 30)
                continue
            mouse_kp, tail_kp = self._split_keypoints_for_e2e(labels_dir, src_out)

            # Step 3: Behavior prediction → runs/end2end/expN/behavior/
            self.set_progress(pct_base + 25, f"[{src_idx+1}/{len(sources)}] Behavior prediction...")
            self._params["mouse_keypoints"] = mouse_kp
            self._params["tail_keypoints"] = tail_kp
            self._params["output_path"] = str(behavior_dir / f"{src_name}_behavior.csv")
            self._run_behavior_inference()

        self._params["pose_inference_sources"] = orig_sources

        # ── Clean up intermediate caches ────────────────────────────
        # E2E inference is one-shot; cached data and normalizers are
        # never reused (paths differ per run).  Delete them.
        import hashlib as _hl, json as _jl, os as _os2
        _cache_root = ".\\dataset_cache"
        for _src_idx in range(len(sources)):
            _mouse_dir = str(pose_dir / f"src_{_src_idx}" / "source_0" / "mouse_labels")
            _tail_dir  = str(pose_dir / f"src_{_src_idx}" / "source_0" / "tail_labels")
            _cfg = {"Mouse_files": [_mouse_dir], "Tail_files": [_tail_dir],
                    "max_instances": 2, "method_version": "v1.9"}
            _h = _hl.md5(_jl.dumps(_cfg, sort_keys=True).encode()).hexdigest()
            for _f in (f"cached_data_{_h}.pt", f"feature_norm_{_h}.npz"):
                _p = _os2.path.join(_cache_root, _f)
                if _os2.path.exists(_p):
                    _os2.remove(_p)
                    self.log(f"Cleaned cache: {_p}", 20)

        self.set_progress(100, f"End-to-End complete → {e2e_run_dir}")

    # ── Behavior Prediction inference ───────────────────────────────

    def _run_behavior_inference(self):
        """Run SABER 3-stage pipeline — delegates to inference.py InferencePipeline.

        Uses the EXACT same code path as the CLI (inference.py) to guarantee
        identical results: same data loading, factor computation, standardization,
        multi-mouse handling, temporal model parameters, and output format.
        """
        params = self._params
        run_dir = params.get("run_dir", "")
        mouse_kp = params.get("mouse_keypoints", "")
        tail_kp = params.get("tail_keypoints", "")
        output_path = params.get("output_path", "results.csv")

        if not run_dir:
            self.error.emit("Behavior Run Dir not specified")
            return
        run_path = Path(run_dir)
        if not run_path.exists():
            self.error.emit(f"Run directory not found: {run_dir}")
            return

        self.log(f"Run directory: {run_dir}", 20)
        self.log(f"Mouse keypoints: {mouse_kp or 'N/A'}", 20)

        # ── Load config (same as CLI: merged_config.yaml) ──
        import json as _json, yaml as _yaml
        merged_yaml = run_path / "configs" / "merged_config.yaml"
        if merged_yaml.exists():
            with open(merged_yaml, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
            self.log("  Config loaded from merged_config.yaml", 20)
        else:
            built_cfg = run_path / "configs" / "built_config.json"
            if built_cfg.exists():
                with open(built_cfg, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                self.log("  Config loaded from built_config.json", 20)
            else:
                self.error.emit("No config found in run/configs/")
                return

        self.set_progress(10, "Initializing inference pipeline...")

        # ── Use the CLI's InferencePipeline directly (guarantees parity) ──
        from inference.inference import (
            InferencePipeline, load_inference_data,
            determine_mouse_order, build_keypoints_for_center,
        )

        pipeline = InferencePipeline(run_path, cfg)

        label_map = cfg.get("label_map", {})
        fps = cfg.get("dataset_config", [{}])[0].get("fps", 30)

        self.set_progress(15, "Loading keypoint data...")
        keypoints_full, flat_attributes, merged_data, normalizer = load_inference_data(
            str(Path(mouse_kp)), str(Path(tail_kp)) if tail_kp else str(Path(mouse_kp)),
            max_instances_num=cfg.get("dataset_config", [{}])[0].get("max_instances_num", 2),
            fps=fps,
        )
        total_frames = keypoints_full.shape[0]
        self.log(f"Keypoint data: {total_frames} frames, D={keypoints_full.shape[1]}", 20)

        # ── Determine mouse order (same heuristic as CLI) ──
        max_instances = cfg.get("dataset_config", [{}])[0].get("max_instances_num", 2)
        center_order = determine_mouse_order(merged_data, max_instances)

        # ── Run inference for each mouse ──
        all_labels = {}
        all_label_names = {}
        all_probas = {}

        for logical_id in range(max_instances):
            actual_center_id = center_order[logical_id]
            self.log(f"  Mouse {logical_id + 1} (center_id={actual_center_id})...", 20)

            progress_base = 20 + logical_id * 35
            self.set_progress(progress_base, f"Running inference for Mouse {logical_id + 1}...")

            kp_center, flat_attrs_center = build_keypoints_for_center(
                keypoints_full, merged_data,
                center_id=actual_center_id,
                max_instance_num=max_instances,
                num_keypoints=10, fps=fps,
                normalizer=normalizer,
            )
            if kp_center.size == 0:
                self.log(f"  Mouse {logical_id + 1}: no valid data, skipping", 30)
                all_labels[logical_id] = np.array([], dtype=int)
                all_label_names[logical_id] = []
                continue

            skip_temporal = self._params.get("skip_temporal", False)
            proba = pipeline.predict(kp_center, flat_attrs_center,
                                     center_id=actual_center_id,
                                     skip_temporal=skip_temporal)
            labels = pipeline.decode_to_labels(proba)
            all_probas[logical_id] = proba
            all_labels[logical_id] = labels

            label_names = [pipeline.id_to_name.get(int(l), f"class_{l}") for l in labels]
            all_label_names[logical_id] = label_names

        # ── Save CSV (match CLI format with probability columns) ──
        self.set_progress(90, "Saving results...")
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-mouse frame count (each mouse has its own independent keypoints)
        n_frames_per_mouse = max((len(v) for v in all_labels.values()), default=0)
        time_sec = np.arange(n_frames_per_mouse, dtype=np.float32) / fps
        id_to_name = pipeline.id_to_name

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["frame", "time_sec"]
            for cid in sorted(all_labels.keys()):
                header += [f"label_mouse{cid + 1}", f"behavior_mouse{cid + 1}"]
            for cid in sorted(all_probas.keys()):
                for ci, cname in enumerate(pipeline.class_names):
                    if ci < all_probas[cid].shape[1]:
                        header.append(f"proba_mouse{cid + 1}_{cname}")
            writer.writerow(header)

            for t in range(n_frames_per_mouse):
                row = [t, round(float(time_sec[t]), 4)]
                for cid in sorted(all_labels.keys()):
                    if t < len(all_labels[cid]):
                        row += [int(all_labels[cid][t]), str(all_label_names[cid][t])]
                    else:
                        row += ["", ""]
                for cid in sorted(all_probas.keys()):
                    for ci in range(all_probas[cid].shape[1]):
                        row.append(round(float(all_probas[cid][t, ci]), 6))
                writer.writerow(row)

        n_frames = sum(len(v) for v in all_labels.values())
        self.log(f"Output saved: {out_path} ({n_frames} frames total)", 20)

        # ── Raster plot (thin strip style, fixed total width) ──
        raster_paths = []
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # Per-mouse direct arrays — each mouse gets its own row
            # with all its frames, no concatenation/splitting needed.
            active_mice = sorted(all_labels.keys())
            n_active = len(active_mice)
            if n_active > 0:
                class_names = [id_to_name.get(i, f"class_{i}") for i in sorted(id_to_name)]
                n_classes = len(class_names)
                cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)

                # Fixed total width: 8 inches regardless of frame count.
                # More frames → higher density (narrower per-frame width).
                strip_h = 0.5  # inches per row
                legend_h = 0.45
                total_h = strip_h * n_active + legend_h
                fig, axes = plt.subplots(n_active, 1,
                                         figsize=(8, total_h),
                                         sharex=True, squeeze=False,
                                         gridspec_kw={'hspace': 0.20, 'top': 0.96,
                                                      'bottom': legend_h/total_h,
                                                      'left': 0.0, 'right': 1.0})
                for row_idx, mi in enumerate(active_mice):
                    ax = axes[row_idx][0]
                    yp = np.asarray(all_labels[mi], dtype=int)
                    if len(yp) == 0:
                        continue
                    colors_arr = cmap([c / max(1, n_classes - 1) for c in range(n_classes)])
                    img = np.zeros((1, len(yp), 3))
                    for c in range(n_classes):
                        img[0, yp == c] = colors_arr[c][:3]
                    ax.imshow(img, aspect='auto', interpolation='nearest')
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.margins(0)
                patches = [mpatches.Patch(color=cmap(i), label=class_names[i])
                           for i in range(n_classes)]
                fig.legend(handles=patches, loc="lower center",
                           ncol=n_classes, fontsize=9, frameon=False,
                           handlelength=1.2, handleheight=1.2, columnspacing=0.5,
                           markerscale=1.0, borderpad=0.1, labelspacing=0.2)
                # Save raster alongside the CSV output (per-source unique name)
                _csv_path = Path(self._params.get("output_path", "results.csv"))
                _raster_stem = _csv_path.stem  # e.g. "C1-M2_behavior"
                rp = str(_csv_path.parent / f"{_raster_stem}_raster.png")
                fig.savefig(rp, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                raster_paths.append(rp)
                self.log(f"Raster plot saved: {rp} ({len(active_mice)} mice, "
                         f"max {max(len(all_labels[mi]) for mi in active_mice)} frames)", 20)
        except Exception as _e:
            self.log(f"Raster plot skipped: {_e}", 30)

        n_frames = sum(len(v) for v in all_labels.values())
        # Build compact label data for E2E live display
        e2e_labels = {}
        for cid in sorted(all_labels.keys()):
            lbls = all_labels[cid]
            if len(lbls) > 0:
                e2e_labels[str(cid)] = [int(x) for x in lbls]
        class_names = [id_to_name.get(i, f"class_{i}") for i in sorted(id_to_name)]

        self.set_progress(100, f"Inference complete — {n_frames} frames")
        self.result_ready.emit({
            "success": True,
            "n_frames": n_frames,
            "raster_plots": raster_paths,
            "output_path": str(out_path),
            "e2e_labels": e2e_labels,
            "e2e_class_names": class_names,
            "message": f"Inference complete: {n_frames} frames → {out_path}",
        })

    @Slot(str, int)
    def _on_worker_log(self, msg, level):
        self.log_line.emit(msg, level)
