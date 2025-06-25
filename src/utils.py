# src/reid_utils.py
import cv2
import os

def save_crop(frame, bbox, track_id, output_dir="outputs/crops"):
    os.makedirs(output_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    filename = f"{output_dir}/player_{track_id}.jpg"
    cv2.imwrite(filename, crop)
