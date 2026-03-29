import pandas as pd
import numpy as np
import os
from pymatgen.core import Element

# -----------------------------
# 1. Polymer compositions (FIXED)
# -----------------------------
polymer_compositions = {
    "PAN": {"C":3, "H":3, "N":1},
    "PEO": {"C":2, "H":4, "O":1},
    "PVDF": {"C":2, "H":2, "F":2},
    "PLA": {"C":3, "H":4, "O":2},
    "PMMA": {"C":5, "H":8, "O":2}
}

# -----------------------------
# 2. Generate 20 systems
# -----------------------------
polymer_systems = []
for p in polymer_compositions:
    for i in range(1,5):
        polymer_systems.append(f"{p}_var{i}")

# -----------------------------
# 3. Experimental conditions (5)
# -----------------------------
conditions = [
    {"temp": 298, "salt": 0.1},
    {"temp": 323, "salt": 0.15},
    {"temp": 348, "salt": 0.2},
    {"temp": 373, "salt": 0.25},
    {"temp": 398, "salt": 0.3}
]

# -----------------------------
# 4. Base mechanical values
# -----------------------------
poisson_base = {"PAN":0.36, "PEO":0.38, "PVDF":0.31, "PLA":0.34, "PMMA":0.37}
anisotropy_base = {"PAN":0.5, "PEO":0.8, "PVDF":0.6, "PLA":0.4, "PMMA":0.5}

# -----------------------------
# 5. Feature functions (FIXED)
# -----------------------------
def heteroatom_ratio(comp):
    total = sum(comp.values())
    hetero = sum(v for k, v in comp.items() if k not in ["C","H"])
    return hetero / total

def electronegativity_diff(comp):
    en = [Element(el).X for el in comp.keys() if Element(el).X is not None]
    return max(en) - min(en)

def polar_atom_ratio(comp):
    polar = ["O","N","F"]
    total = sum(comp.values())
    count = sum(comp.get(el,0) for el in polar)
    return count / total

# -----------------------------
# 6. Dataset generation
# -----------------------------
data = []

for system in polymer_systems:
    polymer = system.split("_")[0]
    comp = polymer_compositions[polymer]

    # Composition-based (constant per polymer)
    hetero_ratio = heteroatom_ratio(comp)
    en_diff = electronegativity_diff(comp)
    polar_ratio = polar_atom_ratio(comp)

    for cond in conditions:
        T = cond["temp"]
        salt = cond["salt"]

        # Mechanical properties
        nu = poisson_base[polymer] + np.random.uniform(-0.01, 0.01)
        A = anisotropy_base[polymer] + np.random.uniform(-0.1, 0.1)

        # Ionic features
        li_fraction = np.random.uniform(0.05, 0.2)
        D = np.random.uniform(1e-11, 1e-10)  # FIXED (no tiny values)

        kB = 1.38e-23
        q = 1.6e-19
        n = li_fraction * 1e28
        sigma = (n * q**2 * D) / (kB * T)

        # Ensure conductivity not zero
        sigma = max(sigma, 1e-6)

        # Stability indices
        energy_above_hull = np.random.uniform(0.01, 0.2)
        TSI = -energy_above_hull
        CSI = -(en_diff * hetero_ratio)

        # Flexibility (proxy)
        flexibility_index = np.random.uniform(0.3, 0.9)

        data.append({
            "polymer_system": system,
            "polymer": polymer,
            "temperature_K": T,
            "salt_concentration": salt,
            "poisson_ratio": round(nu,4),
            "elastic_anisotropy": round(A,4),
            "ionic_conductivity": sigma,
            "li_fraction": li_fraction,
            "electronegativity_difference": en_diff,
            "heteroatom_ratio": hetero_ratio,
            "polar_atom_ratio": polar_ratio,
            "flexibility_index": flexibility_index,
            "thermodynamic_stability_index": TSI,
            "chemical_stability_index": CSI
        })

# -----------------------------
# 7. DataFrame
# -----------------------------
df = pd.DataFrame(data)

# -----------------------------
# 8. Save file (SAFE)
# -----------------------------
file_path = os.path.join(os.getcwd(), "FINAL_polymer_dataset.xlsx")
df.to_excel(file_path, index=False)

# -----------------------------
# 9. Verification
# -----------------------------
print("✅ Dataset generated successfully!")
print("📁 File location:", file_path)
print("📊 Rows:", len(df))

print("\n🔍 Check zeros:")
print((df == 0).sum())
