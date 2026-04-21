import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

print("Loading datasets...")

df_cheating = pd.read_csv("dataset/cheating_behavior.csv")
df_audio = pd.read_csv("dataset/audio_features.csv")
df_eye = pd.read_csv("dataset/eye_tracking.csv")
df_face = pd.read_csv("dataset/face_recognition.csv")
df_expression = pd.read_csv("dataset/facial_expression.csv")
df_object = pd.read_csv("dataset/object_detection.csv")
df_pose = pd.read_csv("dataset/pose_movement.csv")

print(f"cheating_behavior: {df_cheating.shape}")
print(f"audio_features: {df_audio.shape}")
print(f"eye_tracking: {df_eye.shape}")
print(f"face_recognition: {df_face.shape}")
print(f"object_detection: {df_object.shape}")
print(f"pose_movement: {df_pose.shape}")


print("\nCleaning datasets...")

#Audio Features (removeing audio class as it is categorical &not text)
audio_drop = ["audio_class"]
df_audio_clean = df_audio.drop(columns=audio_drop, errors="ignore")

# Eye Tracking
eye_drop = ["gaze_direction"]
df_eye_clean = df_eye.drop(columns=eye_drop, errors="ignore")

# Face Recognition
emb_cols = [c for c in df_face.columns if c.startswith("emb_")]
df_face = df_face.drop(columns=emb_cols + ["identity_id", "registered_id"], errors="ignore")

#Facial Expression
expr_drop = ["emotion_label"]
df_expression_clean = df_expression.drop(columns=expr_drop, errors="ignore")

#Object Detection
df_object_agg = (
    df_object
    .rename(columns={"image_id": "sample_id"})
    .groupby("sample_id")
    .agg(
        total_objects_detected=("annotation_id", "count"),
        suspicious_object_count=("is_suspicious_object", "sum"),
        avg_detection_confidence=("confidence", "mean"),
        phone_detected_flag=("object_class",
                             lambda x: int((x == "phone").any()))
    )
    .reset_index()
)

#Pose Movement
pose_drop = ["movement_label"]
df_pose_clean = df_pose.drop(columns=pose_drop, errors="ignore")

#Cheating Behavior
cheating_keep = [
    "sample_id", "cheating_score", "label_id", "cheating_label",
    "gaze_away_count", "gaze_away_total_sec", "head_turn_count",
    "head_turn_max_angle", "phone_detected_count", "multiple_faces_count",
    "audio_anomaly_count", "whisper_detected_count", "tab_switch_count",
    "fullscreen_exit_count", "identity_mismatch_flag",
    "typing_pause_count", "answer_change_count", "session_duration_sec"
]
df_cheating_clean = df_cheating[cheating_keep]

#Merge
print("\nMerging all datasets on sample_id...")

df_merged = df_cheating_clean.copy()

merge_map = {
    "audio": (df_audio_clean,"sample_id"),
    "eye": (df_eye_clean,"sample_id"),
    "face_recog":(df_face, "sample_id"),
    "expression":(df_expression_clean,"sample_id"),
    "object":(df_object_agg,"sample_id"),
    "pose":(df_pose_clean,"sample_id"),
}

for name, (df, key) in merge_map.items():
    df_prefixed= df.add_prefix(f"{name}_")
    df_prefixed= df_prefixed.rename(columns={f"{name}_{key}": key})
    df_merged =df_merged.merge(df_prefixed, on=key, how="left")
    print(f" Merged {name:12s}=shape now:{df_merged.shape}")


# Missing Values
print("\nHandling missing values...")
nan_counts = df_merged.isnull().sum()
print(f"Total NaN cells before:{nan_counts.sum()}")

# Strategy: fill numeric NaN with column median (robust to outliers)
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
df_merged[numeric_cols] = df_merged[numeric_cols].fillna(df_merged[numeric_cols].median())

print(f"Total NaN cells after : {df_merged.isnull().sum().sum()}")


#Feature
print("\nFeature engineering...")

df_merged["gaze_away_ratio"] = (
    df_merged["gaze_away_total_sec"] / df_merged["session_duration_sec"].replace(0, 1)
)

#suspicion score
df_merged["obj_suspicion_ratio"] = ( df_merged["object_suspicious_object_count"] /
df_merged["object_total_objects_detected"].replace(0, 1)
)



#Define Feature Matrix (X) and Labels (y)
print("\nPreparing final feature matrix...")

# Columns we do not want as input features
non_feature_cols = [
    "sample_id", "cheating_score", "label_id",
    "cheating_label",         
    "audio_class_id",
    "expression_emotion_id",
]

# Build X (features) by dropping non-feature columns
feature_cols = [c for c in df_merged.columns if c not in non_feature_cols]

# Drop any remaining object (text) columns
feature_cols = [c for c in feature_cols
                if df_merged[c].dtype != object]

X = df_merged[feature_cols]
y_class = df_merged["label_id"]          # for classification
y_score = df_merged["cheating_score"]    # for regression

print(f"  Feature matrix X : {X.shape}")
print(f"  Class labels y   : {y_class.value_counts().to_dict()}")


#  Scale Features (Normalisation)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=feature_cols
)

# Save scaler so we can reuse it in the app backend
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  💾 Scaler saved to models/scaler.pkl")

# Also save the list 
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("  💾 Feature columns saved to models/feature_cols.pkl")


#Save Processed Dataset
df_merged.to_csv("processed_data.csv", index=False)
X_scaled["label_id"]      = y_class.values
X_scaled["cheating_score"] = y_score.values
X_scaled.to_csv("features_scaled.csv", index=False)

print("\n✅ Preprocessing complete!")
print("   → processed_data.csv    (full merged dataset)")
print("   → features_scaled.csv   (scaled features + labels, used by train_test.py)")
print("   → models/scaler.pkl     (saved scaler)")
print("   → models/feature_cols.pkl")
