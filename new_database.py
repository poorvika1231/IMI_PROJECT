import pandas as pd
import numpy as np

# Load data
df = pd.read_excel("FINAL_polymer_dataset.xlsx")

# Separate numeric & non-numeric
numeric_df = df.select_dtypes(include=['float64', 'int64'])
non_numeric_df = df.select_dtypes(exclude=['float64', 'int64'])

# Standardize (VERY IMPORTANT)
X = (numeric_df - numeric_df.mean()) / numeric_df.std()

# Orthogonalization using QR decomposition (better than Gram-Schmidt)
Q, R = np.linalg.qr(X.values)

# Convert back to dataframe with SAME column names
numeric_clean = pd.DataFrame(Q, columns=numeric_df.columns)

# Combine back
df_clean = pd.concat([non_numeric_df.reset_index(drop=True),
                      numeric_clean.reset_index(drop=True)], axis=1)

# Save
df_clean.to_excel("FINAL_dataset_NO_REDUNDANCY_ALL_FEATURES.xlsx", index=False)

print("✅ Done")
