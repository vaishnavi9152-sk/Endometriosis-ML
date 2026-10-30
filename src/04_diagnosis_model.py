import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    def build_model(pos_weight):
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False
        )
    MODEL_NAME = "XGBoost"
    print("Using XGBoost")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def build_model(pos_weight):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    MODEL_NAME = "GradientBoosting"
    print("Using GradientBoosting (install xgboost for better results)")

os.makedirs("data/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


df = pd.read_csv("data/structured_endometriosis_data.csv")
print(f"Loaded: {df.shape[0]} patients, {df.shape[1]} columns")
print(f"Diagnosis: {df['Diagnosis'].sum()} positive ({df['Diagnosis'].mean()*100:.1f}%)")
print(f"Columns: {df.columns.tolist()}")


FEATURES = ["Age", "Menstrual_Irregularity", "Chronic_Pain_Level",
            "Hormone_Level_Abnormality", "Infertility", "BMI"]
TARGET = "Diagnosis"

X = df[FEATURES]
y = df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/diagnosis_scaler.joblib")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
model = build_model(pos_weight)
model.fit(X_tr, y_tr)
print(f"{MODEL_NAME} trained!")

y_prob = model.predict_proba(X_te)[:, 1]

THRESHOLD = 0.35
y_pred = (y_prob >= THRESHOLD).astype(int)

print(f"\nClassification Report (threshold={THRESHOLD}):")
print(classification_report(y_te, y_pred,
      target_names=["No endo", "Endo"]))

auc = roc_auc_score(y_te, y_prob)
brier = brier_score_loss(y_te, y_prob)
print(f"ROC-AUC:     {auc:.4f}")
print(f"Brier Score: {brier:.4f}  (lower is better, 0=perfect, 0.25=random)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    build_model(pos_weight), X_scaled, y,
    cv=cv, scoring="roc_auc"
)
print(f"5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

imp = model.feature_importances_
feat_imp = pd.DataFrame({"feature": FEATURES, "importance": imp})
feat_imp = feat_imp.sort_values("importance", ascending=False)

print("\nFeature importances:")
for _, r in feat_imp.iterrows():
    bar = "█" * int(r["importance"] * 300)
    print(f"  {r['feature']:<30} {r['importance']:.4f}  {bar}")

fpr, tpr, _ = roc_curve(y_te, y_prob)
prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=10)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC curve
axes[0].plot(fpr, tpr, color="#1D9E75", linewidth=2.5,
             label=f"Endo diagnosis model (AUC={auc:.3f})")
axes[0].plot([0,1],[0,1],"k--", linewidth=1, label="Random (AUC=0.5)")
axes[0].fill_between(fpr, tpr, alpha=0.08, color="#1D9E75")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title(f"ROC Curve — Endo Diagnosis Prediction\n{MODEL_NAME}  |  n=10,000")
axes[0].legend(fontsize=11)
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])

# Calibration curve
axes[1].plot(prob_pred, prob_true, marker="o", color="#1D9E75",
             linewidth=2, label="Model calibration")
axes[1].plot([0,1],[0,1],"k--", linewidth=1, label="Perfect calibration")
axes[1].fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="#D85A30")
axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("True probability (fraction positive)")
axes[1].set_title(f"Calibration Curve\nBrier Score: {brier:.4f}  (lower=better)")
axes[1].legend(fontsize=10)
axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1])

plt.suptitle("Endometriosis Diagnosis Prediction  |  No Surgery Required",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/plots/06_diagnosis_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: data/plots/06_diagnosis_roc.png")

# Feature importance bar
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1D9E75" if v == feat_imp["importance"].max()
          else "#B4B2A9" for v in feat_imp["importance"]]
ax.bar(feat_imp["feature"], feat_imp["importance"], color=colors)
for i, (_, r) in enumerate(feat_imp.iterrows()):
    ax.text(i, r["importance"] + 0.003, f"{r['importance']:.3f}",
            ha="center", fontsize=10)
ax.set_ylabel("Feature Importance")
ax.set_title("Which Symptoms Best Predict Endometriosis?\n(Green = top predictor)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("data/plots/07_diagnosis_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/plots/07_diagnosis_features.png")

joblib.dump(model, "models/diagnosis_model.joblib")
with open("models/diagnosis_features.json", "w") as f:
    json.dump(FEATURES, f)

print(f"\nSaved: models/diagnosis_model.joblib")
print(f"Saved: models/diagnosis_features.json")

print(f"""
==============================================
DIAGNOSIS MODEL RESULTS  ({MODEL_NAME})
==============================================
ROC-AUC (test):        {auc:.3f}
5-Fold CV AUC:         {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
Brier Score:           {brier:.4f}
Threshold used:        {THRESHOLD}

KEY FINDING FOR YOUR PITCH:
  From 6 clinical features measurable in a GP office
  (age, pain level, hormones, BMI, infertility) we
  predict endo diagnosis with {auc:.0%} AUC.

  This represents the VISION: once we have surgical
  diagnosis labels matched to symptom data, we can
  build a pre-surgical screening tool.

Top predictor: {feat_imp.iloc[0]['feature']}
==============================================
""")
print("Run next: python src/05_multimodal.py")
