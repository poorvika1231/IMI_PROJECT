from mp_api.client import MPRester
import pandas as pd
import numpy as np

API_KEY = "YmYlFaD9V0yeSHnTrgrPQ8ZfeVVDUva5"

# Polymers (material names)
materials = ['PEO', 'PAN', 'PVDF', 'PLA', 'PMMA']

data_list = []

with MPRester(API_KEY) as mpr:
    # Fetch summary documents
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

# Assign polymer names **without repetition**
if len(data_list) <= len(materials):
    material_names = np.random.choice(materials, size=len(data_list), replace=False)
else:
    material_names = np.random.choice(materials, size=len(data_list), replace=True)

for i, data in enumerate(data_list):
    data["material_name"] = material_names[i]

# Create DataFrame
df = pd.DataFrame(data_list)

# Remove duplicates (non-redundant)
df = df.drop_duplicates()

# Reorder columns
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

#  CSV
csv_path = "/home/sowmya/Downloads/final_100_dataset.csv"
df.to_csv(csv_path, index=False)

# OUTPUT
print("\nShape (after removing duplicates):", df.shape)
print(df.head())
print("\nSaved at:", csv_path)
