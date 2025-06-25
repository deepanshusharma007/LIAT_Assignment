# ⚽ Liat AI – Player Re-Identification Assignment

## 📌 Overview
This project performs player and ball re-identification in football match videos using:
- **YOLOv11** for object detection
- **TorchReID** for appearance embedding
- **Agglomerative clustering** for global ID assignment

---

## Project Structure
```
LIAT_Assignment/
├── models/
│   └── best.pt                # ⚠️ NOT included in repo — downloadable separately
│
├── videos/
│   └── 15sec_input_720p.mp4   # Input match video
│
├── outputs/
│   ├── reid_output_offline.mp4     # Final output video (offline re-ID)
│   └── player_id_tracking.csv      # CSV log of global IDs per frame
│
├── src/
│   ├── detector.py            # YOLOv11-based object detector
│   ├── embedder.py            # TorchReID embedding extractor
│   ├── tracker.py             # Tracker + global identity manager
│   ├── sort.py                # Optional SORT fallback (not used in offline ReID)
│   └── utils.py               # Helper utilities for rendering, logging
│
├── .gitignore                 # Ignores venv, outputs, large files
├── requirements.txt           # All Python dependencies
├── README.md                  # Setup instructions + run guide
├── report.md                  # Methodology, experiments, challenges
├── main.py                    # Pipeline entry point
└── .gitattributes             # (optional) Git LFS config if still tracked
```
---

## ⚙️ Setup & Execution (5 Steps Only)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/deepanshusharma007/LIAT_Assignment.git
   cd LIAT_Assignment

2. **Create a virtual environment**:
   ```bash
   python -m venv venv

3. **Activate the virtual environment**:
   ```bash
   **On Windows**
   .\venv\Scripts\activate
   **On MAC**
   source venv/bin/activate

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

5. **Download model**:
   ```
   Create a folder name models inside your project "LIAT_Assignment" and then download the model (.pt) file and put it in models folder
   Here is the model download link : https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

   I attached the project structure also, you can check from the top.

5. **Run the pipeline**:
   ```bash
   python main.py
