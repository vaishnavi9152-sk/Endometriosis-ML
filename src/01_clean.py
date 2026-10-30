
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)


df = pd.read_excel("data/psychometric_data.xlsx", sheet_name="Sheet1")
print(f"Loaded: {df.shape[0]} patients, {df.shape[1]} columns")


before = (df == -77).sum().sum()
df = df.replace(-77, np.nan)
print(f"Replaced {before} occurrences of -77 with NaN")

df["intercourse_pain_reported"] = (~df["intercourse_pain"].isna()).astype(int)
print(f"intercourse_pain_reported: {df['intercourse_pain_reported'].sum()} answered, "
      f"{(df['intercourse_pain_reported']==0).sum()} not applicable")

df["dysmenorrhoea"]      = df["dysmenorrhoea"].fillna(df["dysmenorrhoea"].median())
df["intercourse_pain"]   = df["intercourse_pain"].fillna(0)
df["migrant_background"] = df["migrant_background"].fillna(df["migrant_background"].mode()[0])

print(f"Missing after imputation: {df.isnull().sum().sum()}")

pain_cols = ["dysmenorrhoea", "abdomen_pain", "intercourse_pain", "dysuria", "dyschezia"]
df["pain_burden"]       = df[pain_cols].mean(axis=1).round(2)
df["high_psych_burden"] = ((df["depression"] >= 4) & (df["anxiety"] >= 4)).astype(int)
df["multi_site_pain"]   = ((df["dysuria"] >= 4) & (df["dyschezia"] >= 4)).astype(int)

print(f"pain_burden: mean={df['pain_burden'].mean():.2f}, max={df['pain_burden'].max():.1f}")
print(f"high_psych_burden: {df['high_psych_burden'].sum()} patients ({df['high_psych_burden'].mean()*100:.1f}%)")
print(f"multi_site_pain:   {df['multi_site_pain'].sum()} patients ({df['multi_site_pain'].mean()*100:.1f}%)")

df.to_csv("data/psychometric_clean.csv", index=False)
print(f"\nSaved: data/psychometric_clean.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
print("Run next: python src/02_clustering.py")
