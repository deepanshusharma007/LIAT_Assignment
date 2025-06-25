import numpy as np
from scipy.spatial.distance import cosine

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class PlayerTracker:
    def __init__(self, embedder, max_cosine_distance=0.4, max_age=60):
        self.embedder = embedder
        self.next_id = 1
        self.tracks = {}  # id: {bbox, embeddings[], age}
        self.max_cosine_distance = max_cosine_distance
        self.max_age = max_age

    def update_tracks(self, detections, frame):
        new_tracks = []
        updated_ids = set()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue

            emb = self.embedder.extract(crop[:, :, ::-1])
            emb = emb / np.linalg.norm(emb)

            matched_id = None
            min_dist = float("inf")

            for track_id, track in self.tracks.items():
                avg_emb = np.mean(track['embeddings'], axis=0)
                dist = cosine(emb, avg_emb)
                box_iou = iou((x1, y1, x2, y2), track['bbox'])
                if dist < min_dist and (dist < self.max_cosine_distance and box_iou > 0.05):
                    min_dist = dist
                    matched_id = track_id

            if matched_id:
                self.tracks[matched_id]['bbox'] = (x1, y1, x2, y2)
                self.tracks[matched_id]['embeddings'].append(emb)
                if len(self.tracks[matched_id]['embeddings']) > 5:
                    self.tracks[matched_id]['embeddings'] = self.tracks[matched_id]['embeddings'][-10:]
                self.tracks[matched_id]['age'] = 0
                self.tracks[matched_id]['cls'] = cls
                self.tracks[matched_id]['conf'] = conf
                updated_ids.add(matched_id)
                new_tracks.append((matched_id, (x1, y1, x2, y2), cls, conf))
            else:
                track_id = self.next_id
                self.tracks[track_id] = {'bbox': (x1, y1, x2, y2), 'embeddings': [emb], 'age': 0, 'cls': cls, 'conf': conf}
                self.next_id += 1
                updated_ids.add(track_id)
                new_tracks.append((track_id, (x1, y1, x2, y2), cls, conf))

        expired = []
        for track_id in self.tracks:
            if track_id not in updated_ids:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    expired.append(track_id)

        for track_id in expired:
            del self.tracks[track_id]

        return new_tracks