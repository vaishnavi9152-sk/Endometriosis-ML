import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/tcga_final_ml(Sheet1).csv")


# Drop useless columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
df = df.drop(columns=["submitter_id"], errors="ignore")

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

print("Cleaned Columns:")
print(df.columns)

plt.figure()
df["Age"].hist()
plt.title("Age Distribution (TCGA)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/tcga_age.png")
plt.show()

plt.figure()
plt.scatter(df["Age"], df["BMI"])
plt.title("BMI vs Age (TCGA)")
plt.xlabel("Age")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("data/tcga_bmi_age.png")
plt.show()

features = [
    "Hormone_Level_Abnormality",
    "Menstrual_Irregularity",
    "Infertility",
    "risk_factor_burden",
    "genomic_instability_flag"
]

df[features].hist()
plt.suptitle("Key Clinical Feature Distributions")
plt.tight_layout()
plt.savefig("data/tcga_features.png")
plt.show()

corr = df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr)
plt.colorbar()
plt.title("TCGA Feature Correlation")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()
plt.savefig("data/tcga_heatmap.png")
plt.show()

print("Graphs saved successfully.")