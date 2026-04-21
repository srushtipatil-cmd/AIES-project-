# ProctorAI — AI-Powered Online Exam Proctoring System

ProctorAI is a machine learning-based online exam proctoring system that monitors students in real time during assessments. It detects suspicious behaviour using multimodal signals — gaze tracking, audio analysis, facial expressions, object detection, and pose estimation — and presents live risk scores to an invigilator dashboard.

---

## Features

**Student Side**
- Joins an exam using a unique exam code issued by the teacher
- Takes a timed multiple-choice exam in the browser
- Behaviour events (tab switches, gaze-away, audio anomalies, etc.) are silently logged and sent to the backend

**Teacher / Invigilator Side**
- Creates an exam session and receives a shareable exam code
- Live dashboard showing every student's real-time cheating risk score and severity level (low / medium / high / critical)
- Alert feed showing the most recent suspicious events with timestamps

**Machine Learning Pipeline**
- Trained on seven sensor datasets (audio, eye tracking, face recognition, facial expression, object detection, pose movement, behavioural labels)
- Random Forest Classifier predicts the cheating level: *honest*, *minor cheating*, *moderate cheating*, or *severe cheating*
- Random Forest Regressor produces a continuous risk score (0–100)
- Falls back to a rule-based weighted scorer when models are not yet trained

---

## Project Structure

```
ProctorAI/
│
├── dataset/                        # Raw CSV sensor datasets (not included in repo)
│   ├── cheating_behavior.csv
│   ├── audio_features.csv
│   ├── eye_tracking.csv
│   ├── face_recognition.csv
│   ├── facial_expression.csv
│   ├── object_detection.csv
│   └── pose_movement.csv
│
├── models/                         # Saved model artefacts (auto-created after training)
│   ├── classifier.pkl              # Random Forest Classifier
│   ├── regressor.pkl               # Random Forest Regressor
│   ├── scaler.pkl                  # StandardScaler fitted on training data
│   └── feature_cols.pkl            # Ordered list of feature column names
│
├── plots/                          # Evaluation plots (auto-created after training)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── regression_scatter.png
│
├── data_preprocessing.py           # Step 1 — Load, clean, merge, engineer features
├── train_test.py                   # Step 2 — Train, evaluate, and save ML models
├── app.py                          # Step 3 — Flask REST API backend
├── index.html                      # Frontend (student exam UI + teacher dashboard)
│
├── processed_data.csv              # Full merged dataset (auto-created)
├── features_scaled.csv             # Scaled feature matrix + labels (auto-created)
│
└── requirements.txt                # Python dependencies
```

---

## Machine Learning Architecture

### Data Pipeline (`data_preprocessing.py`)
1. Loads seven raw CSV datasets and merges them on `sample_id`
2. Drops non-numeric and irrelevant columns (embedding vectors, categorical labels, identity IDs)
3. Aggregates the object-detection dataset (one detection per row → one row per sample)
4. Fills missing values with per-column medians
5. Engineers two additional features: `gaze_away_ratio` and `obj_suspicion_ratio`
6. Scales all features using `StandardScaler` and saves the scaler for inference

### Model Training (`train_test.py`)
| Model | Target | Algorithm |
|---|---|---|
| Classifier | `label_id` (0–3 cheating level) | Random Forest (100 trees, max depth 10) |
| Regressor | `cheating_score` (0–100) | Random Forest (100 trees, max depth 10) |

Evaluation outputs: accuracy, classification report, 5-fold cross-validation score, confusion matrix, feature importance chart, and actual-vs-predicted scatter plot.

### Backend Inference (`app.py`)
When a proctoring event arrives from the browser:
1. The event type is mapped to a feature-count column and the running tally is updated
2. The feature vector is built, scaled with the saved `StandardScaler`, and passed to the classifier
3. The predicted class index is mapped to a base score (0 → 10, 1 → 35, 2 → 65, 3 → 88) and fine-tuned using prediction confidence
4. If models are not loaded, a rule-based weighted scorer is used instead

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/login` | Student or teacher login |
| `POST` | `/api/create_test` | Teacher creates an exam session |
| `POST` | `/api/join_exam` | Student joins an exam |
| `GET` | `/api/get_questions` | Returns exam questions |
| `POST` | `/api/proctor_event` | Student browser sends a behaviour event |
| `GET` | `/api/monitor` | Teacher fetches all student risk scores |
| `GET` | `/api/alerts` | Teacher fetches the recent alert feed |
| `POST` | `/api/submit_exam` | Student submits answers; returns exam score |
| `GET` | `/api/health` | Server health check |

---

## Cheating Event Types and Weights

| Event | Weight |
|---|---|
| Identity mismatch | 20 |
| Phone detected | 15 |
| Multiple faces | 12 |
| Whisper detected | 10 |
| Audio speech anomaly | 8 |
| Tab switch | 8 |
| Fullscreen exit | 6 |
| Gaze away | 5 |
| Head turned | 5 |
| Posture suspicious | 4 |
| Hand to face | 3 |

Severity is mapped from the final 0–100 score: **low** (<25), **medium** (25–49), **high** (50–69), **critical** (≥70).

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | Python, pandas, NumPy, scikit-learn |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualisation | matplotlib, seaborn |
| Model Storage | pickle |

---

## Authors

Mini Project Submission — *[Your Name / Team Name]*  
*[Institution Name, Department, Year]*
