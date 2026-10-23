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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
"""
from pathlib import Path

print(list(Path("..").rglob("fattyacids_carbon_34.xlsx"))) #allows for finding the exact directory of files
"""
#reading and importing excel collumn containing the SMILE strings 
df = pd.read_excel('../data/excel/fatty_acids/fattyacids_carbon_34.xlsx') 
mylist = df['smiles_reordered'].tolist()
#print(mylist)

smiles_list_fig3 = np.array (mylist)


#Making the Tanimoto matrix
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

#print Tanimoto matrix for figure 3
fig, ax = subplots(figsize=(12, 8))
ax.axis('off')
table = ax.table(cellText = tanimoto_df_rounded.values, rowLabels = tanimoto_df_rounded.index ,colLabels = tanimoto_df_rounded.columns, loc = 'center', cellLoc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig('tanimoto_table_4.pdf', bbox_inches='tight', dpi=300)
plt.show()