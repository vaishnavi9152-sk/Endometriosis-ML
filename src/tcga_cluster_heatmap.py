import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("data/tcga_final_ml(Sheet1).csv")

# Clean
df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False, na=False)]
df = df.drop(columns=["submitter_id"], errors="ignore")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.median(numeric_only=True))

# Features
features = [
    "Age",
    "BMI",
    "Hormone_Level_Abnormality",
    "Menstrual_Irregularity",
    "Infertility",
    "risk_factor_burden",
    "genomic_instability_flag",
    "neoplasm_histologic_grade",
    "vital_status_flag"
]

X = df[features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels

# Mean feature values per cluster
cluster_means = df.groupby("Cluster")[features].mean()

# Plot heatmap
plt.figure(figsize=(10, 4))
plt.imshow(cluster_means, aspect="auto")
plt.colorbar(label="Mean value")
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(cluster_means.index)), [f"Cluster {i}" for i in cluster_means.index])
plt.title("TCGA Cluster Feature Heatmap")
plt.tight_layout()
plt.savefig("data/tcga_cluster_heatmap.png")
plt.show()

print("\n=== Cluster Means ===")
print(cluster_means)