import argparse
import os
import json
import cv2

class LabelValidator:
    def __init__(self, video_path, annotation_path=None):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Default annotation path if not provided
        base, _ = os.path.splitext(video_path)
        self.annotation_path = annotation_path or base + "_video.json"
        if not os.path.exists(self.annotation_path):
            raise RuntimeError(f"Annotation file not found: {self.annotation_path}")

        # Load annotations
        with open(self.annotation_path, "r") as f:
            self.data = json.load(f)
        self.annotations = []

        for entry in self.data["annotations"]:
            frame_idx = entry["frame_idx"]
            ann = entry["annotation"].split()

            status = ann[0]  # S, I, or V
            if status == "V":
                cx, cy, w, h = map(int, ann[1:])
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                self.annotations.append({
                    "frame_idx": frame_idx,
                    "status": status,
                    "bbox": [x1, y1, x2, y2]
                })
            else:
                self.annotations.append({
                    "frame_idx": frame_idx,
                    "status": status,
                    "bbox": None
                })


        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.idx = 0
        self.window_name = "Validation Tool"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _load_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _draw_annotation(self, frame, annotation):
        state = annotation.get("status")
        bbox = annotation.get("bbox")
        print(f"[DEBUG] Frame {self.idx}: state={state}, bbox={bbox}")

        if state == "V" and bbox:
            # bbox is already [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"V", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif state == "S":
            cv2.putText(frame, "Skipped (S)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif state == "I":
            cv2.putText(frame, "Invisible (I)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def run(self):
        while True:
            if self.idx < 0:
                self.idx = 0
            if self.idx >= self.num_frames:
                self.idx = self.num_frames - 1

            frame = self._load_frame(self.idx)
            if frame is None:
                break

            # Draw annotation if exists
            if self.idx < len(self.annotations):
                annotation = self.annotations[self.idx]
                frame = self._draw_annotation(frame, annotation)

            # Display
            txt = f"Frame [{self.idx+1}/{self.num_frames}]"
            cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                self.idx += 1
            elif key == ord("s"):
                self.idx += 10
            elif key == ord("p"):
                self.idx -= 1
            elif key == ord("b"):
                self.idx -= 10

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True, help="Path to video file.")
    parser.add_argument("--annotations", "-a", default=None, help="Path to annotation JSON file.")
    return parser.parse_args()


def main():
    args = parse_args()
    validator = LabelValidator(args.video, args.annotations)
    validator.run()


if __name__ == "__main__":
    main()
