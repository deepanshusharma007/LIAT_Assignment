# 📄 LIAT AI – Player Re-Identification Assignment Report

## 👇 Problem Statement

To build a robust solution for **player re-identification** in football match videos that ensures:
- Each player is assigned a **unique and consistent ID**
- IDs remain the same even after players leave and re-enter the scene
- The solution should ideally support real-time processing (online)

---

## 🧩 Approach & Methodology

We implemented an **online player re-identification pipeline** using:

### ✅ YOLOv11 for Detection
- A fine-tuned YOLOv11 model (`best.pt`) is used to detect players and the ball frame-by-frame.
- Only class IDs `0` (players) and `1` (ball) are retained; referees are excluded.

### ✅ TorchReID for Appearance Embedding
- Each detected player is cropped and embedded using `osnet_x1_0` from the TorchReID library.
- Features are extracted on-the-fly using padded and normalized crops.

### ✅ Custom Online Tracker
- We maintain a dictionary of active tracks with:
  - Last known bounding box
  - A queue of historical embeddings (rolling memory)
- Re-identification is performed using a hybrid:
  - **Cosine similarity** on embeddings
  - **IoU-based spatial matching**
- IDs are assigned **live**, without waiting for the full video.

### ✅ Final Rendering
- Bounding boxes are drawn with consistent IDs.
- A CSV log (`player_id_tracking.csv`) records every player ID and position per frame.
- The output video is saved as `outputs/reid_output.mp4`.

---

## 🧪 Techniques Tried

| Technique | Outcome |
|----------|---------|
| YOLOv11 + SORT | IDs changed frequently with occlusion |
| YOLOv11 + TorchReID + SORT | Better, but not consistent for overlaps |
| **YOLOv11 + TorchReID + Online Tracker (✅ Final)** | Works in real time, mostly stable, modular |

---

## 🔍 Challenges Encountered

- Player appearance changes due to occlusion or motion blur affected matching.
- Missed detections led to identity switches.
- Highly similar jersey colors (e.g., same team) increased ReID confusion.
- GitHub's 100 MB limit forced `best.pt` to be externally hosted.

---

## 📉 Known Limitations

- Tracking still fails if two players overlap or cross rapidly.
- Ball ID is not fixed since it appears very similar across frames.
- Referees are filtered but sometimes misclassified as players.
- ReID embeddings may drift over long sequences without reset.

---

## 🚀 Future Improvements

### 🧠 AI-Based Enhancements

1. **Train Custom ReID Models**  
   Use football-specific datasets (SoccerNet-v2) to improve ReID accuracy for team uniforms and positions.

2. **Switch to Offline Clustering**  
   For non-real-time use cases, global clustering (e.g., Agglomerative Clustering on all embeddings) can produce more stable global IDs.

3. **Use Graph Neural Networks or Video Transformers**  
   These can reason across time and space, maintaining identity better in long occlusions or camera switches.

4. **Pose Estimation + Color Histograms**  
   Enhance embeddings with color + pose vectors to resolve identity in ambiguous cases.

---

## 📂 What Remains (If More Time Was Available)

- Add jersey number recognition using OCR as a fallback.
- Implement track recovery using lost identity management.
- Fine-tune the ReID embedder on match-specific player crops.
- Deploy as a real-time stream processor using Flask or FastAPI.

---


## ✅ Summary

This project implements a real-time player and ball re-identification system using YOLOv11 + TorchReID + a custom online tracker. It produces stable IDs for many players and maintains modular, scalable architecture. With additional training and smarter association logic, it can achieve production-grade reliability.

---
