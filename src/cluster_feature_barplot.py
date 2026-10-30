import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv(
    "data/Dataset_ROXWELL_Baseline_N699_Meyrose et al.csv",
    encoding="utf-16",
    sep=";"
)

# Drop unnecessary columns
df = df.drop(columns=["ID", "cluster"], errors="ignore")

# Clean values
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Select features
features = [
    "MenstrualBleeding",
    "SexualIntercourse",
    "NRS_01_rec",
    "NRS_02_rec",
    "NRS_03_rec",
    "NRS_04_rec",
    "NRS_05_rec",
    "PDI_total_mean",
    "PHQ_2_total_sum",
    "GAD_2_total_sum",
    "CSQ_Katastr_sum"
]

df_cluster = df[features].copy()
df_cluster = df_cluster.fillna(df_cluster.median(numeric_only=True))

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# Cluster
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df_cluster["Cluster"] = labels

# Mean values
cluster_means = df_cluster.groupby("Cluster").mean().T

# 🔥 RELABEL FEATURES HERE
label_mapping = {
    "MenstrualBleeding": "Actively menstruating",
    "SexualIntercourse": "Sexually active",
    "NRS_01_rec": "Dysmenorrhea (menstrual pain)",
    "NRS_02_rec": "Abdomen pain",
    "NRS_03_rec": "Sexual intercourse pain",
    "NRS_04_rec": "Dysuria (urination pain)",
    "NRS_05_rec": "Dyschezia (bowel pain)",
    "PDI_total_mean": "Pain disability",
    "PHQ_2_total_sum": "Depression (PHQ)",
    "GAD_2_total_sum": "Anxiety (GAD)",
    "CSQ_Katastr_sum": "Pain catastrophizing"
}

cluster_means.index = cluster_means.index.map(label_mapping)

# Plot
cluster_means.plot(kind="bar", figsize=(12, 7))
plt.title("Average Clinical Feature Values by Cluster")
plt.xlabel("Clinical Features")
plt.ylabel("Mean Value")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Cluster")

plt.tight_layout()
plt.savefig("data/cluster_feature_barplot_labeled.png")
plt.show()