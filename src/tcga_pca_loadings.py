import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("data/tcga_final_ml(Sheet1).csv")

# Clean
df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False, na=False)]
df = df.drop(columns=["submitter_id"], errors="ignore")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.median(numeric_only=True))

# Features used in clustering
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

# PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=["PC1", "PC2"]
)

print("\n=== PCA LOADINGS ===")
print(loadings)

print("\n=== Strongest contributors to PC1 ===")
print(loadings["PC1"].abs().sort_values(ascending=False))

print("\n=== Strongest contributors to PC2 ===")
print(loadings["PC2"].abs().sort_values(ascending=False))

print("\n=== Explained Variance ===")
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.3f}")