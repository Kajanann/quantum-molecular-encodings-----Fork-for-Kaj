import sys

sys.path.append("../")
import pandas as pd
import numpy as np
from rdkit import Chem
from quantum_molecular_encodings.supporting_functions import coulomb_matrix, matrix_to_circuit
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
import pandas as pd
from pandas import DataFrame
from numpy.random import rand
from matplotlib.pyplot import subplots
"""
from pathlib import Path

print(list(Path("..").rglob("fattyacids_carbon_34.xlsx"))) #allows for finding the exact directory of files
"""

df = pd.read_excel('../data/excel/fatty_acids/fattyacids_carbon_34.xlsx') 
mylist = df['smiles_reordered'].tolist()
#print(mylist)

smiles_list_fig3 = np.array (mylist)

# Convert SMILES strings to RDKit molecules
molecules = np.array([Chem.MolFromSmiles(smile) for smile in smiles_list_fig3])


# Find the molecule with the largest number of atoms
num_qubits = max(mol.GetNumAtoms() for mol in molecules if mol is not None)

# Generate fingerprints for each molecule
fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
# fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, includeChirality=True, fpSize=2048)
fingerprints = ([fpgen.GetFingerprint(mol) for mol in molecules])

# Calculate Tanimoto similarity matrix
n = len(fingerprints)
tanimoto_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            tanimoto_matrix[i][j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
        else:
            tanimoto_matrix[i][j] = 1.0  # Similarity with itself is 1

# Display the Tanimoto similarity matrix
tanimoto_df = pd.DataFrame(tanimoto_matrix, index=smiles_list_fig3, columns=smiles_list_fig3)
tanimoto_df_rounded = tanimoto_df.round(6)
