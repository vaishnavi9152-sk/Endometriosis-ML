
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

os.makedirs("data/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/psychometric_clean.csv")
print(f"Loaded: {df.shape[0]} patients")

features = [
    "dysmenorrhoea",        # menstrual pain        (0-10)
    "abdomen_pain",         # pelvic pain           (0-10)
    "intercourse_pain",     # pain during sex       (0-10)
    "dysuria",              # bladder pain          (0-10)
    "dyschezia",            # bowel pain            (0-10)
    "pain_disability",      # pain limiting life    (0-70)
    "depression",           # PHQ score             (0-6)
    "anxiety",              # GAD score             (0-6)
    "pain_catastrophising", # catastrophising       (6-35)
]
X = df[features].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/kmeans_scaler.joblib")


print("\nTesting k=2 to k=7:")
scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels)
    scores[k] = s
    print(f"  k={k}  silhouette={s:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nBest k by math: {best_k}  |  We use k=4 for clinical interpretability")

# --- PLOT: silhouette scores ---
fig, ax = plt.subplots(figsize=(7, 4))
ks   = list(scores.keys())
vals = list(scores.values())
cols = ["#D85A30" if k == 4 else "#B4B2A9" for k in ks]
bars = ax.bar(ks, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette Score by k  —  Higher is Better")
ax.set_ylim(0, max(vals) + 0.06)
plt.tight_layout()
plt.savefig("data/plots/01_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/plots/01_silhouette.png")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, "models/kmeans_model.joblib")

print("\nCluster sizes:")
for c in sorted(df["cluster"].unique()):
    n = (df["cluster"] == c).sum()
    print(f"  Cluster {c}: {n} patients ({n/len(df)*100:.1f}%)")


print("\nANOVA — are clusters significantly different?")
for feat in features:
    groups = [df[df["cluster"] == c][feat].values for c in range(4)]
    _, p = f_oneway(*groups)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NS"
    print(f"  {feat:<25}  p={p:.2e}  {sig}")


outcome_cols = ["major_dep", "anxiety_disorder", "work_ability", "pain_burden"]
profile_cols = features + [c for c in outcome_cols if c not in features]
profile = df.groupby("cluster")[profile_cols].mean().round(2)

print("\nCluster means (key features):")
show = ["abdomen_pain", "dysuria", "dyschezia", "pain_disability",
        "depression", "anxiety", "pain_catastrophising"]
print(profile[show].to_string())

print("\nClinical outcome rates:")
print(f"{'Cluster':<10} {'n':<7} {'Major dep%':<13} {'Anxiety dis%':<15} {'Cant work%':<12} {'Pain burden'}")
print("-" * 65)
for c in range(4):
    sub = df[df["cluster"] == c]
    print(f"  {c:<8} {len(sub):<7} "
          f"{sub['major_dep'].mean()*100:<13.1f} "
          f"{sub['anxiety_disorder'].mean()*100:<15.1f} "
          f"{(sub['work_ability']==2).mean()*100:<12.1f} "
          f"{sub['pain_burden'].mean():.2f}")


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
ev = pca.explained_variance_ratio_

COLORS = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517"]
NAMES  = [
    "High pain + high psych burden",
    "Low burden (resilient)",
    "Multi-site physical pain",
    "High pain + coping",
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: PCA scatter
for c in range(4):
    mask = df["cluster"] == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=COLORS[c], alpha=0.6, s=28,
                    label=f"Cluster {c} — {NAMES[c]} (n={mask.sum()})")
axes[0].set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)")
axes[0].set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)")
axes[0].set_title("4 Patient Subtypes — PCA Projection\n(699 real endo patients, ROXWELL study)")
axes[0].legend(fontsize=8, loc="upper right")

# Right: depression + anxiety rates per cluster
dep_rates = [df[df["cluster"]==c]["major_dep"].mean()*100 for c in range(4)]
anx_rates = [df[df["cluster"]==c]["anxiety_disorder"].mean()*100 for c in range(4)]
x = np.arange(4)
w = 0.35
axes[1].bar(x - w/2, dep_rates, w, color=[c + "CC" for c in COLORS], label="Major depression %")
axes[1].bar(x + w/2, anx_rates, w, color=COLORS, alpha=0.55, label="Anxiety disorder %")
axes[1].axhline(8, color="gray", linestyle="--", linewidth=1, label="General population (~8%)")
for i, (d, a) in enumerate(zip(dep_rates, anx_rates)):
    axes[1].text(i - w/2, d + 1, f"{d:.0f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].text(i + w/2, a + 1, f"{a:.0f}%", ha="center", fontsize=9, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"Cluster {c}" for c in range(4)], fontsize=10)
axes[1].set_ylabel("% of patients")
axes[1].set_title("Depression & Anxiety Rates by Cluster\nvs General Population Baseline")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 105)

plt.suptitle("Endometriosis Patient Subtype Discovery  |  ROXWELL Study n=699",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/plots/02_cluster_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/plots/02_cluster_pca.png")

plot_feats  = ["abdomen_pain", "dysmenorrhoea", "dysuria",
               "dyschezia", "pain_disability", "depression",
               "anxiety", "pain_catastrophising"]
plot_labels = ["Abdomen\npain", "Menstrual\npain", "Bladder\npain",
               "Bowel\npain", "Pain\ndisability", "Depression",
               "Anxiety", "Catastrophising"]
maxvals = df[plot_feats].max().values

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(plot_feats))
w = 0.2
for i, c in enumerate(range(4)):
    means = profile.loc[c, plot_feats].values.astype(float) / maxvals
    ax.bar(x + i*w, means, w, color=COLORS[c], alpha=0.85,
           label=f"Cluster {c}: {NAMES[c]}")
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(plot_labels, fontsize=9)
ax.set_ylabel("Normalized score (0 = min, 1 = max)")
ax.set_title("Feature Profile by Patient Subtype  |  4 Clusters, 699 Patients")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("data/plots/03_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/plots/03_cluster_profiles.png")

df.to_csv("data/psychometric_clustered.csv", index=False)
print(f"\nSaved: data/psychometric_clustered.csv")
print("Run next: python src/03_depression_model.py")
