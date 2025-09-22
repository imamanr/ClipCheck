from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import cv2
import torch
import numpy as np
import os
from pathlib import Path

class PromptBasedAnnotator:
    def __init__(self, processor, model, conf_thresh=0.1):
        """
        Args:
            processor: HuggingFace processor (e.g., from transformers)
            model: HuggingFace model (e.g., GroundingDINO or similar)
            conf_thresh: confidence threshold for filtering boxes
        """
        self.prompt_detector_processor = processor
        self.prompt_detector_model = model
        self.conf_thresh = conf_thresh

    def annotate_frame(self, frame, prompt, output_format="txt"):
        """
        Run prompt-based detection on a single frame.

        Args:
            frame (np.ndarray): OpenCV image (BGR format)
            prompt (str): text prompt for detection
            output_format (str): "txt" (V/S/I format) or "dict"

        Returns:
            str or dict: annotation result
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run through processor + model
        inputs = self.prompt_detector_processor(
            text=[prompt], images=rgb_frame, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.prompt_detector_model(**inputs)

        # Get logits and boxes
        logits = outputs.logits.softmax(-1)[0]          # take first image in batch
        boxes = outputs.pred_boxes[0]                     # take first image in batch

        class_id = 0  # for "pedestrian", or look up dynamically
        scores = logits[:, class_id]
        best_idx = scores.argmax().item()

        # Pick highest scoring box
        best_box = boxes[best_idx].detach().cpu().numpy()  # xywh normalized
        best_score = scores[best_idx].item()        

        if best_score < self.conf_thresh:
            # No confident detection
            if output_format == "txt":
                return "S -1 -1 -1 -1"
            else:
                return {"state": "S", "bbox": None, "score": best_score}

        # Denormalize box
        h, w = frame.shape[:2]
        cx, cy, bw, bh = best_box
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h

        if output_format == "txt":
            return int(cx), int(cy), int(bw), int(bh)
        else:
            return {
                "state": "V",
                "bbox": [int(cx), int(cy), int(bw), int(bh)],
            }

# from autodistill.detection import CaptionOntology
# from autodistill_grounded_sam import GroundedSAM
# import supervision as sv
# from autodistill_clip import CLIP

# import os
# from pathlib import Path
# import json
# import numpy as np
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# from autodistill.core.composed_detection_model import ComposedDetectionModel
# import cv2

# classes = ["person"]

# class promptOntology:
#     def __init__(self, classes):
#         self.classes = classes
#     def annotate_frame(self, frame, prompt, output_format="txt"):
#         SAMCLIP = ComposedDetectionModel(
#             detection_model=GroundedSAM(
#                 CaptionOntology({
#                     "person": [
#                 prompt
#             ],
#             })),
#             classification_model=CLIP(
#                 CaptionOntology({k: k for k in classes})
#             )
#         )

#         results = SAMCLIP.predict(frame)
#         bbox= [-1, -1, -1, -1]
#         if results.confidence.any():
#             best_idx = results.confidence.argmax()

#             # Get best confidence score
#             best_conf = results.confidence[best_idx]

#             # Extract best bbox
#             bbox = [
#                 int(results.xyxy[best_idx][0]),
#                 int(results.xyxy[best_idx][1]),
#                 int(results.xyxy[best_idx][2]),
#                 int(results.xyxy[best_idx][3])
#             ]
#         return f'V {int((bbox[0]+bbox[2])/2)} {int((bbox[1]+bbox[3])/2)} {int(bbox[2]-bbox[0])} {int(bbox[3]-bbox[1])}'


if __name__ == "__main__":

    # Example with HuggingFace GroundingDINO (replace with your model)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")

    # Initialize annotator
    annotator = PromptBasedAnnotator(processor, model, conf_thresh=0.35)

    # Load frame
    frame = cv2.imread("street_scene.jpg")

    # Prompt
    annotation = annotator.annotate_frame(frame, "a pedestrian", output_format="txt")
    print(annotation)  # e.g., "V 420 310 60 120"
