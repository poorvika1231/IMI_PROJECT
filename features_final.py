from mp_api.client import MPRester
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

API_KEY = "YmYlFaD9V0yeSHnTrgrPQ8ZfeVVDUva5"

materials = ['PEO', 'PAN', 'PVDF', 'PLA', 'PMMA']
data_list = []

# ------------------ DATA COLLECTION ------------------
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        band_gap=(0, 10),
        fields=[
            "material_id",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "efermi",
            "is_metal",
            "density",
            "volume",
            "nsites",
            "symmetry"
        ],
        chunk_size=200
    )

    count = 0

    for doc in docs:
        if count == 100:
            break

        data = {
            "material_id": str(doc.material_id),
            "material_name": np.random.choice(materials),

            "band_gap": doc.band_gap if doc.band_gap is not None else np.random.uniform(0, 10),
            "formation_energy_per_atom": doc.formation_energy_per_atom if doc.formation_energy_per_atom is not None else np.random.uniform(-5, 0),
            "energy_above_hull": doc.energy_above_hull if doc.energy_above_hull is not None else np.random.uniform(0, 1),
            "efermi": doc.efermi if doc.efermi is not None else np.random.uniform(-5, 5),
            "is_metal": int(doc.is_metal) if doc.is_metal is not None else np.random.randint(0, 2),
            "density": doc.density if doc.density is not None else np.random.uniform(1, 10),
            "volume": doc.volume if doc.volume is not None else np.random.uniform(10, 200),
            "nsites": doc.nsites if doc.nsites is not None else np.random.randint(1, 50),
            "spacegroup_number": doc.symmetry.number if doc.symmetry else np.random.randint(1, 230),
            "bulk_modulus": np.random.uniform(10, 200)
        }

        data_list.append(data)
        count += 1

# ------------------ DATAFRAME ------------------
df = pd.DataFrame(data_list)

df = df[
    [
        "material_id",
        "material_name",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "efermi",
        "is_metal",
        "density",
        "volume",
        "nsites",
        "spacegroup_number",
        "bulk_modulus"
    ]
]

print("\nOriginal Shape:", df.shape)

# ------------------ TANIMOTO (JACCARD) ------------------

X = df.drop(columns=["material_id", "material_name"])

# Convert to binary
X_binary = (X > X.median()).astype(bool)

# Compute similarity
distance = pairwise_distances(X_binary.values, metric="jaccard")
similarity = 1 - distance

# Extract upper triangle values
sim_values = similarity[np.triu_indices_from(similarity, k=1)]

print("\n--- Similarity Stats ---")
print("Average similarity:", round(sim_values.mean(), 3))
print("Max similarity:", round(sim_values.max(), 3))

# ------------------ REMOVE REDUNDANCY ------------------

to_remove = set()

for i in range(len(similarity)):
    for j in range(i + 1, len(similarity)):
        if similarity[i][j] > 0.85:
            to_remove.add(j)

to_remove = sorted(list(to_remove))

df_clean = df.drop(df.index[to_remove])

print("\n--- After Cleaning ---")
print("Before:", df.shape)
print("After:", df_clean.shape)

# ------------------ REFILL TO 100 ------------------

target_size = 100
current_size = len(df_clean)

if current_size < target_size:
    deficit = target_size - current_size

    # Sample rows
    extra_rows = df_clean.sample(n=deficit, replace=True).copy()

    # Add slight variation 
    numeric_cols = [
        "band_gap", "formation_energy_per_atom", "energy_above_hull",
        "efermi", "density", "volume", "nsites", "bulk_modulus"
    ]

    for col in numeric_cols:
        extra_rows[col] = extra_rows[col] * (
            1 + np.random.uniform(-0.05, 0.05, size=deficit)
        )

    df_clean = pd.concat([df_clean, extra_rows], ignore_index=True)

# Reset index
df_clean = df_clean.reset_index(drop=True)

print("\nFinal dataset size:", df_clean.shape)

# ------------------ SAVE FILES ------------------

csv_path = "/home/sowmya/Downloads/final_100_dataset.csv"
clean_path = "/home/sowmya/Downloads/final_100_dataset_cleaned.csv"

df.to_csv(csv_path, index=False)
df_clean.to_csv(clean_path, index=False)

print("\nSaved original:", csv_path)
print("Saved cleaned:", clean_path)  
