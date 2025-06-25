import cv2
import os
from src.detector import PlayerDetector
from src.embedder import ReIDEmbedder
from src.tracker import PlayerTracker

VIDEO_PATH = 'videos/15sec_input_720p.mp4'
OUTPUT_PATH = 'outputs/reid_output.mp4'
os.makedirs('outputs', exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

detector = PlayerDetector('models/best.pt')
embedder = ReIDEmbedder()
tracker = PlayerTracker(embedder)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect_players(frame)
    tracks = tracker.update_tracks(detections, frame)

    for track_id, (x1, y1, x2, y2), cls, conf in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
