from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path='models/best.pt', conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_players(self, frame):
        results = self.model(frame, conf=self.conf)[0]
        # results = self.model(frame, conf=self.conf, augment=True)[0]
        detections = []
        detected_classes = []
        for box in results.boxes:
            cls = int(box.cls[0])
            detected_classes.append(cls)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls in [0, 1, 2, 3]:  # Players, Ball, Referees, Ball (if 3)
                detections.append((x1, y1, x2, y2, conf, cls))
        print(f"[DEBUG] Detected classes in frame: {detected_classes}")
        print(f"[DEBUG] Frame detections: {len(detections)}")
        return detections