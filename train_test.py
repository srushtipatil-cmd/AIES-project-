
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score
)

#load
print("Loading features_scaled.csv ...")

if not os.path.exists("features_scaled.csv"):
    raise FileNotFoundError(
        "features_scaled.csv not found. Please run data_preprocessing.py first!"
    )

df = pd.read_csv("features_scaled.csv")
print(f"  Dataset shape: {df.shape}")

#classifi y
y_class  = df["label_id"]     
y_score  = df["cheating_score"]  

# Drop label columns from features
X = df.drop(columns=["label_id", "cheating_score"], errors="ignore")

print(f"  Features (X) : {X.shape}")
print(f"  Class distribution:\n{y_class.value_counts().sort_index()}")


#Train/Test Split

print("\nSplitting into train / test sets (80% / 20%)")

X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_score,
    test_size=0.2,
    random_state=42 
)

print(f"  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")


#Train Classification Model
print("\n🌲 Training Random Forest Classifier ...")

# Random Forest = many decision trees combined → more accurate, less overfit
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1       
)

clf.fit(X_train, y_train_cls)
print("  ✅ Classifier training complete!")


#Evaluate Classification Model
print("\n📊 Evaluating classifier on test set ...")

y_pred_cls = clf.predict(X_test)

accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"\n  Test Accuracy : {accuracy * 100:.2f}%")

label_names = ["honest", "minor_cheating", "moderate_cheating", "severe_cheating"]
print("\n  Classification Report:")
print(classification_report(y_test_cls, y_pred_cls, target_names=label_names))

# Cross-validation: trains & tests on 5 different splits → more reliable score
cv_scores = cross_val_score(clf, X, y_class, cv=5, scoring="accuracy")
print(f"  5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Confusion Matrix Plot
os.makedirs("plots", exist_ok=True)

cm = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_names, yticklabels=label_names
)
plt.title("Confusion Matrix — Cheating Classification")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()
print("  📈 Confusion matrix saved → plots/confusion_matrix.png")


#  STEP 5: Feature Importance Plot
importances = pd.Series(clf.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 15 Most Important Features (Classifier)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("  📈 Feature importance saved → plots/feature_importance.png")

print("\n  Top 5 most important features:")
for feat, imp in importances.nlargest(5).items():
    print(f"    {feat:45s} : {imp:.4f}")


#  STEP 6: Train Regression Model
print("\n🌲 Training Random Forest Regressor (cheating score 0-100) ...")

reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_train, y_train_reg)
y_pred_reg = reg.predict(X_test)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2  = r2_score(y_test_reg, y_pred_reg)
print(f"  ✅ Regressor training complete!")
print(f"  Mean Absolute Error (MAE) : {mae:.2f} points")
print(f"  R² Score                  : {r2:.4f}  (1.0 = perfect)")

# Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.4, color="steelblue", s=15)
plt.plot([0, 100], [0, 100], "r--", label="Perfect prediction")
plt.xlabel("Actual Cheating Score")
plt.ylabel("Predicted Cheating Score")
plt.title("Actual vs Predicted Cheating Score")
plt.legend()
plt.tight_layout()
plt.savefig("plots/regression_scatter.png", dpi=150)
plt.close()
print("  📈 Regression scatter saved → plots/regression_scatter.png")


#  STEP 7: Save Trained Models
print("\n💾 Saving models ...")

os.makedirs("models", exist_ok=True)

with open("models/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("  ✅ models/classifier.pkl")

with open("models/regressor.pkl", "wb") as f:
    pickle.dump(reg, f)
print("  ✅ models/regressor.pkl")


#  STEP 8: Quick Sanity Check — Predict One Sample
print("\n🔍 Quick sanity check — predicting one sample ...")

# Take the first row from the test set
sample = X_test.iloc[[0]]
actual_label = label_names[int(y_test_cls.iloc[0])]
actual_score = round(y_test_reg.iloc[0], 1)

pred_label = label_names[int(clf.predict(sample)[0])]
pred_score = round(reg.predict(sample)[0], 1)
pred_proba = clf.predict_proba(sample)[0]

print(f"  Actual  → label: {actual_label}, score: {actual_score}")
print(f"  Predict → label: {pred_label}, score: {pred_score}")
print(f"  Class probabilities: { {label_names[i]: round(p,3) for i,p in enumerate(pred_proba)} }")


print("\n" + "="*55)
print("✅ Training and evaluation complete!")
print("="*55)
print("models/classifier.pkl→ cheating level classifier")
print("models/regressor.pkl→ cheating score regressor")
print("plots/→ confusion matrix, feature importance, scatter")
