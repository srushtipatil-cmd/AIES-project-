# Installation & Setup Guide — ProctorAI

Follow these steps in order to get the full ProctorAI system running on your local machine.

---

## Prerequisites

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.9+ | `python --version` |
| pip | 21+ | `pip --version` |
| A modern web browser | Chrome / Firefox / Edge | — |

> **Note:** Python 3.10 or 3.11 is recommended. The project is not tested on Python 3.12+.

---

## Step 1 — Clone or Download the Project

If you have Git installed:
```bash
git clone <repository-url>
cd ProctorAI
```

Or unzip the submitted project folder and open a terminal inside it:
```bash
cd path/to/ProctorAI
```

---

## Step 2 — Create a Virtual Environment (Recommended)

A virtual environment keeps the project dependencies isolated from your system Python.

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

## Step 3 — Install Dependencies

Install all required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install packages manually:
```bash
pip install flask flask-cors pandas numpy scikit-learn matplotlib seaborn
```

### Full Dependency List

| Package | Purpose |
|---|---|
| `flask` | Web framework for the backend API |
| `flask-cors` | Allows the browser frontend to call the API |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and feature vectors |
| `scikit-learn` | Machine learning models, scaler, metrics |
| `matplotlib` | Plotting evaluation charts |
| `seaborn` | Confusion matrix heatmap styling |
| `pickle` (built-in) | Saving and loading trained models |

---

## Step 4 — Add the Dataset Files

Place the following CSV files inside a folder named `dataset/` in the project root:

```
ProctorAI/
└── dataset/
    ├── cheating_behavior.csv
    ├── audio_features.csv
    ├── eye_tracking.csv
    ├── face_recognition.csv
    ├── facial_expression.csv
    ├── object_detection.csv
    └── pose_movement.csv
```

> These files are not included in the repository due to size. Obtain them from the project dataset source or your instructor.

---

## Step 5 — Run the Data Preprocessing Script

This script merges, cleans, and scales all datasets, and saves the scaler and feature columns for later use by the app.

```bash
python data_preprocessing.py
```

**Expected output (abbreviated):**
```
Loading datasets...
cheating_behavior: (N, 20)
...
Merging all datasets on sample_id...
 Merged audio       = shape now: (N, 45)
...
Feature matrix X : (N, 87)
✅ Preprocessing complete!
   → processed_data.csv
   → features_scaled.csv
   → models/scaler.pkl
   → models/feature_cols.pkl
```

After this step you will see two new CSV files in the project root and a `models/` folder containing the scaler.

---

## Step 6 — Train the Machine Learning Models

This script trains the Random Forest Classifier and Regressor, evaluates them, and saves the trained models.

```bash
python train_test.py
```

**Expected output (abbreviated):**
```
🌲 Training Random Forest Classifier ...
✅ Classifier training complete!

Test Accuracy : XX.XX%
5-Fold Cross-Validation Accuracy: XX.XX% ± X.XX%

🌲 Training Random Forest Regressor ...
Mean Absolute Error (MAE) : X.XX points
R² Score                  : 0.XXXX

💾 Saving models ...
  ✅ models/classifier.pkl
  ✅ models/regressor.pkl

✅ Training and evaluation complete!
```

After this step the `models/` folder will contain all four artefacts and a `plots/` folder will contain three evaluation charts.

---

## Step 7 — Start the Flask Backend

```bash
python app.py
```

**Expected output:**
```
=======================================================
  🚀 ProctorAI Backend Server Starting
=======================================================
  ML Model   : ✅ Loaded
  Scaler     : ✅ Loaded
  Frontend   : Serving index.html from current directory
  URL        : http://localhost:5000
=======================================================
 * Running on http://0.0.0.0:5000
```

> If you see `⚠️ Not found` next to the model, re-run Steps 5 and 6.

---

## Step 8 — Open the Application

Open your browser and navigate to:
```
http://localhost:5000
```

The `index.html` frontend is served directly by Flask from the project root.

---

## Using the Application

### As a Teacher
1. Select **Teacher** role on the login screen and enter your name.
2. Click **Create Exam** to generate a unique exam code (e.g. `EXAM-AB12XY`).
3. Share the exam code with your students.
4. Open the **Monitor Dashboard** to see live risk scores and alerts for all connected students.

### As a Student
1. Select **Student** role, enter your name, and enter the exam code given by your teacher.
2. Click **Start Exam** to begin the timed multiple-choice test.
3. The browser will silently monitor behaviour events (gaze, tab switches, etc.) and report them to the backend.
4. Submit your answers when done — your exam score and final integrity status will be displayed.

---

## Verifying the Server is Running

Open a new browser tab or use curl to check the health endpoint:
```
http://localhost:5000/api/health
```

Expected JSON response:
```json
{
  "status": "running",
  "ml_model_loaded": true,
  "active_students": 0,
  "total_alerts": 0,
  "active_exams": 0
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flask` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `FileNotFoundError: features_scaled.csv` | Preprocessing not run | Run `python data_preprocessing.py` first |
| `ML Model: ⚠️ Not found` | Training not run | Run `python train_test.py` after preprocessing |
| `Address already in use` on port 5000 | Another process using port 5000 | Change port in `app.py`: `app.run(port=5001)` |
| `FileNotFoundError: dataset/...csv` | Dataset folder missing | Create `dataset/` folder and add the CSV files |
| Browser shows `CORS error` | Flask-CORS not installed | Run `pip install flask-cors` |

---

## Project Run Order Summary

```
1. pip install -r requirements.txt
2. python data_preprocessing.py
3. python train_test.py
4. python app.py
5. Open http://localhost:5000
```

---

## File Outputs Reference

| File | Created By | Purpose |
|---|---|---|
| `processed_data.csv` | `data_preprocessing.py` | Full merged dataset |
| `features_scaled.csv` | `data_preprocessing.py` | Scaled features + labels for training |
| `models/scaler.pkl` | `data_preprocessing.py` | StandardScaler for inference |
| `models/feature_cols.pkl` | `data_preprocessing.py` | Ordered feature column names |
| `models/classifier.pkl` | `train_test.py` | Trained cheating level classifier |
| `models/regressor.pkl` | `train_test.py` | Trained cheating score regressor |
| `plots/confusion_matrix.png` | `train_test.py` | Classification evaluation chart |
| `plots/feature_importance.png` | `train_test.py` | Top 15 features by importance |
| `plots/regression_scatter.png` | `train_test.py` | Actual vs predicted score chart |
