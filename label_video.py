import argparse
import os, sys, json
import cv2
import numpy as np
from pathlib import Path
from YOLOTracker import YOLOTracker
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PromptBasedAnnotator import PromptBasedAnnotator

class LabelData:
    def __init__(self):
        self.shapes = []
        self.imageHeight = 0
        self.imageWidth = 0
        self.state = None

    def to_labelme(self):
        if self.state == "S":
            return "S -1 -1 -1 -1"
        elif self.state == "I":
            return "I -1 -1 -1 -1"
        elif self.state == "V" and self.shapes:
            s = self.shapes[0]
            (x1, y1), (x2, y2) = s["points"]
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            return f"V {int(x_center)} {int(y_center)} {int(w)} {int(h)}"
        else:
            return f'V -1 -1 -1 -1'

class CVLabelTool:
    def __init__(self, video_path, annotation_path=None, prompt=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")

        base, _ = os.path.splitext(video_path)
        self.annotation_path = annotation_path or base + "_video.json"
        os.makedirs(os.path.dirname(self.annotation_path), exist_ok=True)

        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.idx = 0
        self.window_name = "cv_labelme"

        self.curr_label = "object"
        self.data = LabelData()
        self.base_img = None
        self.display = None

        self.bbox = None
        self.tracker = None
        self.tracking = False
        self.await_fix = False
        self.manual_override = False

        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.prompt = prompt
        if prompt:
            processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
            self.annotator = PromptBasedAnnotator(processor, model, conf_thresh=0.15)

        self._load_frame()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        # Store all frame labels
        self.video_annotations = []

    def _load_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
        ret, img = self.cap.read()
        if not ret:
            raise RuntimeError("End of video or failed to read frame")
        self.base_img = img
        self.display = img.copy()
        self.data = LabelData()
        h, w = img.shape[:2]
        self.data.imageHeight = h
        self.data.imageWidth = w

    def _mouse_cb(self, event, x, y, flags, param):
        if self.await_fix or (not self.tracking and (not self.bbox or getattr(self, 'manual_override', False))):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.drag_start = (x, y)
                self.drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                self.drag_end = (x, y)
                x0, y0 = self.drag_start
                x1, y1 = self.drag_end
                x_min, x_max = sorted([x0, x1])
                y_min, y_max = sorted([y0, y1])
                w, h = x_max - x_min, y_max - y_min
                bbox = (x_min, y_min, w, h)
                print(f"New bbox drawn: {bbox}")
                self._init_tracker_with_bbox(bbox, label=self.curr_label)
                self.manual_override = False

        self._refresh_display()

    def _refresh_display(self):
        self.display = self.base_img.copy()
        if self.bbox:
            cx, cy, w, h = map(int, self.bbox)
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.data.shapes = [{
                "label": self.curr_label,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],  # store corners
                "shape_type": "bbox"
            }]
        txt = f"[{self.idx+1}/{self.num_frames}] "
        if self.manual_override and not self.bbox:
            txt += "Manual labeling -> Draw bbox with mouse"
        elif not self.bbox:
            txt += "Press L to label"
        elif self.await_fix:
            txt += "Prediction -> (A)ccept / (F)ix"
        else:
            txt += "Tracking..."
        cv2.putText(self.display, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0), 2)

    def detect_with_prompt(self, frame, prompt):
        boxes = self.annotator.annotate_frame(prompt=[prompt], frame=frame)
        if boxes and len(boxes) > 0:
            x, y, bw, bh = boxes
            return int(x), int(y), int(bw), int(bh)
        return None

    def _init_tracker_with_bbox(self, bbox, label=None):
        self.bbox = bbox
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        self.bbox = (cx, cy, w, h) 
        
        self.tracker = YOLOTracker()
        self.tracker.init(self.base_img, (x, y, w, h))
        self.data.state = "V"
        self.data.shapes = [{
            "label": label or self.curr_label,
            "points": [[float(cx), float(cy)], [float(w), float(h)]],
            "shape_type": "bbox"
        }]
        self.tracking = True
        self.await_fix = True
        print(f"Initialized tracker with bbox: {self.bbox}")

    def save_frame_annotation(self):
        ann = self.data.to_labelme()
        self.video_annotations.append({
            "frame_idx": self.idx,
            "annotation": ann
        })

    def run(self):
        self.manual_override = False

        while True:
            if self.prompt and not self.bbox and not self.manual_override:
                print(f"Running prompt-based detection with prompt: {self.prompt}")
                norm_bbox = self.detect_with_prompt(frame=self.base_img, prompt=self.prompt)
                if norm_bbox is not None:
                    self._init_tracker_with_bbox(norm_bbox, label="prompt_obj")
                else:
                    print("Prompt-based detector found nothing.")

            if self.tracking and not self.await_fix:
                ok, newbox = self.tracker.update(self.base_img)
                if ok:
                    x, y, w, h = map(int, newbox)
                    cx, cy = x + w / 2.0, y + h / 2.0
                    self.bbox = (cx, cy, w, h)
                    self.await_fix = True
                    self.data.shapes = [{
                        "label": self.curr_label,
                        "points": [[float(cx), float(cy)], [float(w), float(h)]],
                        "shape_type": "bbox"
                    }]
                else:
                    print("Tracking failed, need relabel (press L).")
                    self.tracking = False
                    self.bbox = None
                    self.data.state = "I"

            self._refresh_display()
            cv2.imshow(self.window_name, self.display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a") and self.await_fix:
                print("Accepted prediction")
                self.data.state = "V"
                self.save_frame_annotation()
                self.idx += 1
                if self.idx < self.num_frames:
                    self._load_frame()
                self.await_fix = False
                
            elif key == ord("s"):
                print("Skipped frame")
                self.data.state = "S"
                self.save_frame_annotation()
                self.idx += 1
                if self.idx < self.num_frames:
                    self._load_frame()
                self.await_fix = False
                
            elif key == ord("l"):
                if self.prompt:
                    print("Re-running prompt-based detector...")
                    self.bbox = None
                    self.tracking = False
                else:
                    print("Draw bbox with mouse")
                    self.manual_override = True
            elif key == ord("f"):
                print("Fallback to manual labeling → draw with mouse")
                self.manual_override = True
                self.await_fix = True
                self.bbox = None
                self.tracking = False

            if self.idx > self.num_frames:
                print("Reached end of video frames.")
                break
        cv2.destroyAllWindows()

        # Save video-level JSON
        video_data = {
            "video_path": self.video_path,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
            "width": self.width,
            "height": self.height,
            "annotations": self.video_annotations
        }
        with open(self.annotation_path, "w") as f:
            json.dump(video_data, f, indent=2)
        print(f"Saved full video annotations to {self.annotation_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", help="Path to video file.")
    parser.add_argument("--out", "-o", default=None, help="Output JSON file for video annotations.")
    parser.add_argument("--prompt", "-p", default=None, help="Prompt for the labeling.")
    return parser.parse_args()

def main():
    args = parse_args()
    tool = CVLabelTool(args.video, args.out, args.prompt)
    tool.run()

if __name__ == "__main__":
    main()
