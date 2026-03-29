import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import random

polymers = {
    "PEO": "CCO",
    "PAN": "C(C#N)C",
    "PVDF": "C(C(F)F)C(F)F",
    "PLA": "CC(C(=O)O)C",
    "PMMA": "CC(C(=O)OC)C",
    "PS": "C(C1=CC=CC=C1)C",
    "PTFE": "C(F)(F)C(F)F",
    "PP": "CC(C)C",
    "PCL": "CCCC(=O)O",
    "PET": "C(C(=O)OCC)C(=O)O",
    "PMGI": "CC(C(=O)O)C(C)C",
    "PVA": "C(CO)O",
    "PVP": "C(C1=CC=CC=N1)C",
    "PBT": "C(C(=O)OCC)C(=O)O",
    "PSU": "C1=CC=CC=C1S(=O)(=O)C",
    "PC": "C(C(=O)OCC)C(=O)O",
    "PLA-PEG": "CC(C(=O)OCC)C(OC)O",
    "PPO": "CC(C)O",
    "PBAT": "CC(C(=O)O)C(=O)O",
    "POM": "C1COC1"
}

data = []
for polymer, smi in polymers.items():
    for salt_conc in [0.1, 0.2, 0.3, 0.4, 0.5]:  
        temperature = random.choice([298, 323])  
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        monomer_weight = Descriptors.MolWt(mol)
        chain_length = random.randint(10, 50)  
        molecular_weight = monomer_weight * chain_length  
        Tg = random.uniform(250, 350)
        Tm = random.uniform(120, 200)
        TPSA_val = CalcTPSA(mol)
        dipole_moment = random.uniform(0.5, 4.0)
        solubility_param = random.uniform(10, 25)
        surface_energy = random.uniform(0.05, 0.2)

        heteroatom_ratio = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [6,1])/mol.GetNumAtoms()
        polar_atom_ratio = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [7,8,9,16])/mol.GetNumAtoms()
        
        flexibility_index = min(chain_length / (molecular_weight/monomer_weight), 1.0)

        logP = Crippen.MolLogP(mol)
        num_H_donors = Lipinski.NumHDonors(mol)
        num_H_acceptors = Lipinski.NumHAcceptors(mol)

        data.append([
            polymer, salt_conc, temperature, Tg, Tm, molecular_weight, chain_length,
            TPSA_val, dipole_moment, solubility_param, surface_energy,
            heteroatom_ratio, polar_atom_ratio, flexibility_index,
            logP, num_H_donors, num_H_acceptors, smi
        ])

columns = [
    "polymer","salt_concentration","temperature","Tg","Tm","molecular_weight",
    "chain_length","TPSA","dipole_moment","solubility_param","surface_energy",
    "heteroatom_ratio","polar_atom_ratio","flexibility_index",
    "logP","num_H_donors","num_H_acceptors","smiles"
]

df = pd.DataFrame(data, columns=columns)
print("Generated dataset shape:", df.shape)

df_clean = df.drop_duplicates(subset=["polymer","salt_concentration","temperature"]).reset_index(drop=True)
print("Cleaned dataset shape:", df_clean.shape)

ranges = {
    "Tg": (200, 400),
    "Tm": (100, 250),
    "molecular_weight": (100, 2500),  
    "chain_length": (10, 50),
    "TPSA": (0, 150),
    "dipole_moment": (0, 5),
    "solubility_param": (5, 30),
    "surface_energy": (0, 0.5),
    "heteroatom_ratio": (0, 1),
    "polar_atom_ratio": (0, 1),
    "flexibility_index": (0, 1),
    "logP": (-2, 6),
    "num_H_donors": (0, 10),
    "num_H_acceptors": (0, 10)
}

for col, (low, high) in ranges.items():
    out_of_range = df_clean[(df_clean[col]<low) | (df_clean[col]>high)]
    print(f"{col}: {len(out_of_range)} rows out of range")
    if len(out_of_range) > 0:
        print(out_of_range[[col,"polymer","smiles"]])

df_clean.to_csv("polymer_features_final_corrected.csv", index=False)
df_clean.to_excel("polymer_features_final_corrected.xlsx", index=False)
print("Saved dataset as CSV and Excel.")
