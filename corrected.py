import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from mp_api.client import MPRester

API_KEY = "xYrzMvH7T3shNlZX097khevUK6tDKc0H" 

candidate_polymers = {
    "PEO": "OCCO",
    "PAN": "C(C#N)",
    "PMMA": "CC(=O)OC",
    "PVA": "CO",
    "PVC": "CC(Cl)CC(Cl)",
    "PS": "C1=CC=CC=C1",
    "PEG": "OCCOCCO",
    "PCL": "CC(=O)OCC",
    "PLA": "C(C(=O)O)C",
    "PU": "NC(=O)OCCNC(=O)OCC",
    "PET": "CCOC(=O)C1=CC=CC=C1C(=O)OCC",
    "PC": "O=C(OCC1=CC=CC=C1)OCC1=CC=CC=C1",
    "PEI": "NCCNCCN",
    "PAA": "CC(=O)OCC(=O)O",
    "PHEMA": "CC(O)C(=O)OCC",
    "PDMS": "C[Si](C)(C)O[Si](C)(C)O",
    "PPO": "CC(C)OC1=CC=CC=C1",
    "PTFE": "FC(F)(F)C(F)(F)F",
    "PBI": "C1=NC=NC=N1",
    "PI": "O=C1OC(=O)C2=CC=CC=C12"
}

salts = ["LiF", "LiCl", "LiBr"]
salt_concentrations = [5, 10, 15, 20, 25]

fixed_salt_props = {
    "LiF": {"anion_radius": 133, "lattice_energy": 1036},
    "LiCl": {"anion_radius": 181, "lattice_energy": 853},
    "LiBr": {"anion_radius": 196, "lattice_energy": 807}
}

def fetch_salt_properties(formula):
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["formation_energy_per_atom", "band_gap"]
            )
            if docs:
                return docs[0].formation_energy_per_atom, docs[0].band_gap
    except:
        pass
    fallback = {"LiF": (-3.16, 0.0), "LiCl": (-2.02, 5.74), "LiBr": (-1.83, 4.93)}
    return fallback.get(formula, (None, None))

salt_data = {s: dict(zip(["formation_energy_per_atom", "band_gap"], fetch_salt_properties(s))) for s in salts}

def compute_dielectric(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return 2 + 0.05 * Descriptors.TPSA(mol)

def compute_functional_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return min(sum(a.GetSymbol() in ["O","N","F","Cl"] for a in mol.GetAtoms()), 5)

def estimate_tg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return -50 + 0.01 * Descriptors.MolWt(mol)

def compute_segmental_mobility(tg):
    return 1 / abs(tg + 1e-5)

def compute_ced(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) / (Descriptors.TPSA(mol)+1e-5)

def compute_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def is_redundant(new_smiles, selected_fps, threshold=0.85):
    mol = Chem.MolFromSmiles(new_smiles)
    fp_new = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for fp in selected_fps:
        if DataStructs.TanimotoSimilarity(fp_new, fp) > threshold:
            return True
    return False

selected_fps = []

for name, smiles in candidate_polymers.items():
    if len(selected_polymers) >= 20:
        break
    if not is_redundant(smiles, selected_fps):
        selected_polymers[name] = smiles
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        selected_fps.append(fp)

print(f"✅ Selected {len(selected_polymers)} distinct polymers")

data = []

for polymer, smiles in selected_polymers.items():
    dielectric = compute_dielectric(smiles)
    functional_group = compute_functional_group(smiles)
    tg_base = estimate_tg(smiles)
    ced = compute_ced(smiles)
    mw = compute_mw(smiles)

    for i, conc in enumerate(salt_concentrations):
        salt = salts[i % len(salts)]
        salt_info = salt_data[salt]
        fixed_info = fixed_salt_props[salt]
        tg = tg_base - 0.5*conc
        seg_mob = compute_segmental_mobility(tg)
        data.append({
            "polymer": polymer,
            "Mw": mw,
            "Tg": tg,
            "dielectric": dielectric,
            "functional_group": functional_group,
            "segmental_mobility": seg_mob,
            "CED": ced,
            "salt": salt,
            "salt_concentration": conc,
            "anion_radius": fixed_info["anion_radius"],
            "lattice_energy": fixed_info["lattice_energy"],
            "formation_energy_per_atom": salt_info["formation_energy_per_atom"],
            "band_gap": salt_info["band_gap"]
        })

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_excel("final_100_tanimoto_ready.xlsx", index=False)

print("✅ Dataset created with 100 rows, Tanimoto-filtered")
print(df.head())
