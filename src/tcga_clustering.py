import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("data/tcga_final_ml(Sheet1).csv")


df = df.loc[:, ~df.columns.str.contains("Unnamed")]
df = df.drop(columns=["submitter_id"], errors="ignore")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.median(numeric_only=True))


features = [
    "Age",
    "BMI",
    "Hormone_Level_Abnormality",
    "Menstrual_Irregularity",
    "Infertility",
    "risk_factor_burden",
    "genomic_instability_flag"
]

X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("Trying different k values...")

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k} -> silhouette: {score:.3f}")


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title("TCGA Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("data/tcga_clusters.png")
plt.show()
summary = df.groupby("Cluster")[features].mean()
print("\nCluster Summary:\n")
print(summary)

print("\nCluster Sizes:\n")
print(df["Cluster"].value_counts())