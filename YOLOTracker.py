import argparse, os, sys, json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO  # pip install ultralytics

def iou(boxA, boxB):
    # box format: x, y, w, h
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    return inter / float(areaA+areaB-inter+1e-6)

class YOLOTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.target_box = None

    def init(self, frame, init_box):
        """Initialize tracker with a manually drawn box"""
        self.target_box = init_box

    def update(self, frame):
        """Run YOLO detection and return best-matching box"""
        results = self.model.predict(frame, verbose=False)[0]
        best_iou = 0
        best_box = None
        for box in results.boxes.xywh:  # xywh format
            x, y, w, h = box.tolist()
            candidate = (int(x-w/2), int(y-h/2), int(w), int(h))
            if self.target_box:
                score = iou(self.target_box, candidate)
                if score > best_iou:
                    best_iou = score
                    best_box = candidate
        if best_box is not None:
            self.target_box = best_box
            return True, best_box #int(cx), int(cy), int(w), int(h)
        return False, None
