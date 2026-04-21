from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
import random
import string
import time
import datetime
import numpy as np

#setup
app = Flask(__name__, static_folder=".")
CORS(app)                                 

# Load

def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

clf= load_model("models/classifier.pkl")
reg= load_model("models/regressor.pkl")
scaler= load_model("models/scaler.pkl")
feature_cols =load_model("models/feature_cols.pkl")

if clf:
    print("Classifier model loaded")
else:
    print("No classifier found — run data_preprocessing.py + train_test.py first")

LABEL_NAMES = ["honest", "minor_cheating", "moderate_cheating", "severe_cheating"]



users    = {} 
exams    = {}  
sessions = {} 
alerts   = [] 


def make_id(prefix="U"):
    """Generate a short unique random."""
    return prefix + "-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

def make_exam_code():
    """Generate a 6-character exam code"""
    return "EXAM-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

def now_ts():
    """Return current ISO timestamp"""
    return datetime.datetime.utcnow().isoformat() + "Z"

def get_severity(score):
    """Map a 0-100 risk score to a severity label."""
    if score >= 70:   return "critical"
    if score >= 50:   return "high"
    if score >= 25:   return "medium"
    return "low"

def compute_cheating_score_from_events(events: list) -> float:
    
    weights = {
        "gaze_away"         : 5,
        "head_turned"       : 5,
        "phone_detected"    : 15,
        "multiple_faces"    : 12,
        "audio_speech"      : 8,
        "whisper_detected"  : 10,
        "tab_switch"        : 8,
        "fullscreen_exit"   : 6,
        "hand_to_face"      : 3,
        "posture_suspicious": 4,
        "identity_mismatch" : 20,
    }
    score = 0.0
    for event in events:
        etype = event.get("type", "")
        score += weights.get(etype, 2) 
    return min(score, 100.0)


def build_feature_vector(event_summary: dict) -> np.ndarray:
    """
    Converts a dict of event counts/values into a numpy array
    that matches the shape the ML model was trained on.
    
    event_summary keys match column names from processed_data.csv
    """
    if feature_cols is None:
        return None

    # Build a row of zeros, then fill in whatever we have
    row = {col: 0.0 for col in feature_cols}

    # Map incoming frontend event data to feature column names
    mapping = {
        "gaze_away_count"       : event_summary.get("gaze_away_count", 0),
        "gaze_away_total_sec"   : event_summary.get("gaze_away_total_sec", 0),
        "head_turn_count"       : event_summary.get("head_turn_count", 0),
        "phone_detected_count"  : event_summary.get("phone_detected_count", 0),
        "multiple_faces_count"  : event_summary.get("multiple_faces_count", 0),
        "audio_anomaly_count"   : event_summary.get("audio_anomaly_count", 0),
        "whisper_detected_count": event_summary.get("whisper_detected_count", 0),
        "tab_switch_count"      : event_summary.get("tab_switch_count", 0),
        "fullscreen_exit_count" : event_summary.get("fullscreen_exit_count", 0),
        "identity_mismatch_flag": event_summary.get("identity_mismatch_flag", 0),
        "typing_pause_count"    : event_summary.get("typing_pause_count", 0),
        "answer_change_count"   : event_summary.get("answer_change_count", 0),
        "session_duration_sec"  : event_summary.get("session_duration_sec", 0),
    }

    for col, val in mapping.items():
        if col in row:
            row[col] = float(val)

    vec = np.array([row[col] for col in feature_cols]).reshape(1, -1)
    return vec

#  ROUTES — Static Files

@app.route("/")
def serve_frontend():
    """Serve the main index.html file."""
    return send_from_directory(".", "index.html")


#  ROUTE: POST /api/login
#  Called when student or teacher logs in

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()

    name = data.get("name", "").strip()
    role = data.get("role", "student") 
    code = data.get("exam_code", "")

    if not name:
        return jsonify({"error": "Name is required"}), 400

    # Create a user entry if this is a new user
    user_id = make_id("T" if role == "teacher" else "S")
    users[user_id] = {"name": name, "role": role}

    if role == "teacher":
        return jsonify({
            "user_id"  : user_id,
            "name"     : name,
            "role"     : "teacher",
            "message"  : f"Welcome, {name}! Create an exam to get started."
        })

    # Student: check exam code
    if code not in exams:
        return jsonify({"error": f"Exam code '{code}' not found"}), 404

    exam = exams[code]
    return jsonify({
        "user_id"   : user_id,
        "name"      : name,
        "role"      : "student",
        "exam_code" : code,
        "exam_title": exam["title"],
        "duration"  : exam["duration"],
        "message"   : f"Welcome, {name}!"
    })




@app.route("/api/create_test", methods=["POST"])
def create_test():
    data = request.get_json()

    teacher_id = data.get("teacher_id", "")
    title      = data.get("title", "ProctorAI Exam")
    duration   = int(data.get("duration", 30))    # minutes

    code = make_exam_code()

    # Default questions (can be extended)
    default_questions = [
        {"id": 1, "q": "What is the time complexity of binary search?",
         "opts": ["O(n)", "O(log n)", "O(n²)", "O(1)"], "ans": 1},
        {"id": 2, "q": "Which data structure uses LIFO ordering?",
         "opts": ["Queue", "Stack", "Heap", "Graph"], "ans": 1},
        {"id": 3, "q": "What does HTTP stand for?",
         "opts": ["HyperText Transfer Protocol", "High Transfer Text Protocol",
                  "Hyper Transfer Technology Protocol", "HyperText Technology Process"], "ans": 0},
        {"id": 4, "q": "Which sorting algorithm has the best average-case complexity?",
         "opts": ["Bubble Sort", "Selection Sort", "Quick Sort", "Insertion Sort"], "ans": 2},
        {"id": 5, "q": "What does SQL stand for?",
         "opts": ["Structured Query Language", "Simple Query Language",
                  "Standard Query Logic", "Stored Query Link"], "ans": 0},
    ]

    exams[code] = {
        "code"      : code,
        "title"     : title,
        "duration"  : duration,
        "teacher_id": teacher_id,
        "questions" : default_questions,
        "created_at": now_ts()
    }

    print(f"📝 Exam created: {code} by teacher {teacher_id}")
    return jsonify({
        "code"     : code,
        "title"    : title,
        "duration" : duration,
        "message"  : "Exam created successfully"
    })



@app.route("/api/join_exam", methods=["POST"])
def join_exam():
    data = request.get_json()

    student_id = data.get("student_id", "")
    exam_code  = data.get("exam_code", "")
    name       = data.get("name", "Student")

    if exam_code not in exams:
        return jsonify({"error": "Invalid exam code"}), 404

    # Initialise student session tracking
    sessions[student_id] = {
        "student_id"        : student_id,
        "name"              : name,
        "exam_code"         : exam_code,
        "exam_started"      : True,
        "exam_submitted"    : False,
        "cheating_score"    : 0.0,
        "alerts_count"      : 0,
        "severity"          : "low",
        "start_time"        : time.time(),
        "events"            : [],  
        "event_counts"      : {    
            "gaze_away_count"       : 0,
            "gaze_away_total_sec"   : 0,
            "head_turn_count"       : 0,
            "phone_detected_count"  : 0,
            "multiple_faces_count"  : 0,
            "audio_anomaly_count"   : 0,
            "whisper_detected_count": 0,
            "tab_switch_count"      : 0,
            "fullscreen_exit_count" : 0,
            "identity_mismatch_flag": 0,
            "typing_pause_count"    : 0,
            "answer_change_count"   : 0,
        }
    }

    print(f"🎓 Student {name} ({student_id}) joined exam {exam_code}")
    return jsonify({"message": f"Joined exam {exam_code}", "exam_code": exam_code})

#  Returns the question list for an exam


@app.route("/api/questions", methods=["GET"])
def get_questions():
    code = request.args.get("exam_code", "")

    if code not in exams:
        return jsonify({"error": "Exam not found"}), 404

    # Return questions WITHOUT the answer key (security!)
    qs = []
    for q in exams[code]["questions"]:
        qs.append({
            "id"  : q["id"],
            "q"   : q["q"],
            "opts": q["opts"]
        })

    return jsonify({"questions": qs, "duration": exams[code]["duration"]})

#  Frontend sends proctoring alerts here in real-time

@app.route("/api/proctor_event", methods=["POST"])
def proctor_event():
    data = request.get_json()

    student_id = data.get("student_id", "unknown")
    event_type = data.get("type", "unknown")       # e.g. "gaze_away", "phone_detected"
    detail     = data.get("detail", "")
    confidence = float(data.get("confidence", 1.0))

    # Ignore low-confidence events
    if confidence < 0.4:
        return jsonify({"message": "Low confidence, ignored"})

    # Get or create session
    if student_id not in sessions:
        sessions[student_id] = {
            "student_id"        : student_id,
            "name"              : data.get("student_name", "Unknown"),
            "exam_code"         : "",
            "exam_started"      : True,
            "exam_submitted"    : False,
            "cheating_score"    : 0.0,
            "alerts_count"      : 0,
            "severity"          : "low",
            "start_time"        : time.time(),
            "events"            : [],
            "event_counts"      : {k: 0 for k in [
                "gaze_away_count","gaze_away_total_sec","head_turn_count",
                "phone_detected_count","multiple_faces_count","audio_anomaly_count",
                "whisper_detected_count","tab_switch_count","fullscreen_exit_count",
                "identity_mismatch_flag","typing_pause_count","answer_change_count"
            ]}
        }

    session = sessions[student_id]

    # Record the event
    event_record = {
        "type"        : event_type,
        "detail"      : detail,
        "confidence"  : confidence,
        "ts"          : now_ts(),
        "student_id"  : student_id,
        "student_name": session["name"],
    }
    session["events"].append(event_record)
    session["alerts_count"] += 1

    # Update running event count (for ML feature vector)
    count_key_map = {
        "gaze_away"         : "gaze_away_count",
        "head_turned"       : "head_turn_count",
        "phone_detected"    : "phone_detected_count",
        "multiple_faces"    : "multiple_faces_count",
        "audio_speech"      : "audio_anomaly_count",
        "whisper_detected"  : "whisper_detected_count",
        "tab_switch"        : "tab_switch_count",
        "fullscreen_exit"   : "fullscreen_exit_count",
        "identity_mismatch" : "identity_mismatch_flag",
    }
    count_key = count_key_map.get(event_type)
    if count_key and count_key in session["event_counts"]:
        session["event_counts"][count_key] += 1

    # Update session duration
    session["event_counts"]["session_duration_sec"] = time.time() - session["start_time"]

    # ── Recompute cheating score ──
    if clf is not None and scaler is not None and feature_cols is not None:
        # Use trained ML model for prediction
        vec = build_feature_vector(session["event_counts"])
        if vec is not None:
            try:
                vec_scaled = scaler.transform(vec)
                pred_class = int(clf.predict(vec_scaled)[0])
                proba      = clf.predict_proba(vec_scaled)[0]
                # Convert class (0-3) to score (0-100)
                class_to_score = {0: 10, 1: 35, 2: 65, 3: 88}
                base_score  = class_to_score[pred_class]
                # Add some variation based on probability confidence
                score = base_score + (proba[pred_class] - 0.5) * 20
                session["cheating_score"] = max(0.0, min(100.0, round(score, 1)))
            except Exception as e:
                print(f"  ⚠️ ML prediction error: {e}")
                session["cheating_score"] = compute_cheating_score_from_events(session["events"])
    else:
        # Fallback: rule-based score
        session["cheating_score"] = compute_cheating_score_from_events(session["events"])

    session["severity"] = get_severity(session["cheating_score"])

    # Decide alert severity
    severity_map = {
        "phone_detected"    : "critical",
        "identity_mismatch" : "critical",
        "multiple_faces"    : "high",
        "whisper_detected"  : "high",
        "gaze_away"         : "medium",
        "tab_switch"        : "medium",
        "head_turned"       : "medium",
        "audio_speech"      : "low",
        "fullscreen_exit"   : "low",
    }
    event_record["severity"] = severity_map.get(event_type, "low")

    # Add to global alerts list (teacher sees these)
    alerts.insert(0, event_record)   # insert at front so newest is first
    if len(alerts) > 500:           # keep only last 500 alerts
        alerts.pop()

    print(f"  🚨 [{session['severity'].upper()}] {session['name']} — {event_type} (score: {session['cheating_score']})")

    return jsonify({
        "message"       : "Event recorded",
        "cheating_score": session["cheating_score"],
        "severity"      : session["severity"],
    })

#  Teacher dashboard — get all student statuses

@app.route("/api/monitor", methods=["GET"])
def monitor():
    teacher_id = request.args.get("teacher_id", "")

    # Get all active student sessions
    student_list = []
    for sid, sess in sessions.items():
        student_list.append({
            "student_id"    : sid,
            "name"          : sess["name"],
            "cheating_score": round(sess["cheating_score"], 1),
            "severity"      : sess["severity"],
            "exam_started"  : sess["exam_started"],
            "exam_submitted": sess["exam_submitted"],
            "alerts_count"  : sess["alerts_count"],
        })

    # Sort by cheating score descending (highest risk first)
    student_list.sort(key=lambda s: s["cheating_score"], reverse=True)

    avg_score = (
        sum(s["cheating_score"] for s in student_list) / len(student_list)
        if student_list else 0
    )
    high_risk = sum(1 for s in student_list if s["cheating_score"] >= 50)

    return jsonify({
        "total_students"    : len(student_list),
        "students"          : student_list,
        "avg_cheating_score": round(avg_score, 1),
        "high_risk_count"   : high_risk,
    })

#  Returns recent alerts for teacher dashboard

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    limit = int(request.args.get("limit", 20))
    return jsonify({"alerts": alerts[:limit]})

#  Student submits their exam

@app.route("/api/submit_exam", methods=["POST"])
def submit_exam():
    data = request.get_json()

    student_id = data.get("student_id", "")
    answers    = data.get("answers", {})   
    exam_code  = data.get("exam_code", "")

    if student_id in sessions:
        sessions[student_id]["exam_submitted"] = True
        sessions[student_id]["exam_started"]   = False

    # Calculate score if we have the answer key
    score = None
    total = None
    if exam_code in exams:
        questions = exams[exam_code]["questions"]
        correct = 0
        for q in questions:
            student_ans = answers.get(str(q["id"]))
            if student_ans is not None and int(student_ans) == q["ans"]:
                correct += 1
        score = correct
        total = len(questions)

    print(f"  📬 {student_id} submitted exam. Score: {score}/{total}")
    return jsonify({
        "message" : "Exam submitted successfully",
        "score"   : score,
        "total"   : total,
        "cheating_score": round(sessions.get(student_id, {}).get("cheating_score", 0), 1)
    })



#  Simple health check — useful for debugging


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status"         : "running",
        "timestamp"      : now_ts(),
        "ml_model_loaded": clf is not None,
        "active_students": len(sessions),
        "total_alerts"   : len(alerts),
        "active_exams"   : len(exams),
    })
#  MAIN — Start the Server

if __name__ == "__main__":
    print("=" * 55)
    print("  🚀 ProctorAI Backend Server Starting")
    print("=" * 55)
    print(f"  ML Model   : {'✅ Loaded' if clf else '⚠️  Not found (run training first)'}")
    print(f"  Scaler     : {'✅ Loaded' if scaler else '⚠️  Not found'}")
    print(f"  Frontend   : Serving index.html from current directory")
    print(f"  URL        : http://localhost:5000")
    print("=" * 55)

    # debug=True → auto-reloads on code change (useful during development)
    app.run(debug=True, host="0.0.0.0", port=5000)
