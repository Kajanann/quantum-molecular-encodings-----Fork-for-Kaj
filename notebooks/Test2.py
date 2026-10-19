# %%
import sys

sys.path.append("../")
# %%
### QMSE scheme ###
import numpy as np
from rdkit import Chem
from quantum_molecular_encodings.supporting_functions import coulomb_matrix, matrix_to_circuit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger 
from PyPDF2 import PdfReader

#%%
### Merges the pdf images generated in the code into one image###
def join_pdf(output_path, *input_path):
    merger = PDFMerger()
    for path in input_paths:
        merger,append(path)
    merger.write(output_path)
    merger.close()
print("Pdf's merged!")


#%%
# Example SMILES string
smiles_string = "CCCC"
molecule_from_smiles = Chem.MolFromSmiles(smiles_string)
num_qubits = molecule_from_smiles.GetNumAtoms()

# Generate the Coulomb matrix without hydrogen atoms
coulomb_matrix_without_h = coulomb_matrix(smiles_string, add_hydrogens=False)

# Encoding circuit #TODO: test with different n_atom_to_qubit values
circuit = matrix_to_circuit(coulomb_matrix_without_h, num_qubits = num_qubits, entangling_layer='rzz', n_atom_to_qubit=1, interleaved='rzz', reverse_bits=True, n_layers=1)
"""
circuit.draw(reverse_bits=True, output='mpl')
plt.savefig("circuit.pdf", format="pdf", bbox_inches='tight')


# %%
from qiskit.visualization import plot_state_qsphere
from qiskit.quantum_info import Statevector

statevector = np.array(Statevector(circuit))
plot_state_qsphere(statevector, figsize=(6, 6)).savefig('sphereplot.pdf')
"""
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
import pandas as pd
from pandas import DataFrame
from numpy.random import rand
from matplotlib.pyplot import subplots

# Define the SMILES strings for the alkanes
smiles_list = np.array(["CC","CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"])
#smiles_list = np.array(["C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "C1CCCCCC1", "C1CCCCCCC1", "C1CCCCCCCC1", "C1CCCCCCCCC1"])
#smiles_list = np.array(["C=C/C=C/CC#N", "C=CC(=C)CC#N"])
#smiles_list = np.array(['c1ccccc1', 'c1c2ccccc2ccc1', 'c1ccc2cc3ccccc3cc2c1', 'c1c2cc3cc4ccccc4cc3cc2ccc1', 'c1c2cc3cc4cc5ccccc5cc4cc3cc2ccc1'])

# Convert SMILES strings to RDKit molecules
molecules = np.array([Chem.MolFromSmiles(smile) for smile in smiles_list])


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
tanimoto_df = pd.DataFrame(tanimoto_matrix, index=smiles_list, columns=smiles_list)
tanimoto_df_rounded = tanimoto_df.round(6)

"""
fig, ax = subplots(figsize=(12, 8))
ax.axis('off')
table = ax.table(cellText = tanimoto_df_rounded.values, rowLabels = tanimoto_df_rounded.index ,colLabels = tanimoto_df_rounded.columns, loc = 'center', cellLoc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig('tanimoto_table_4.pdf', bbox_inches='tight', dpi=300)
plt.show()
"""

### QMSE similarities ###

from quantum_molecular_encodings.supporting_functions import quantum_overlap

overlap_df_x_xx = quantum_overlap(num_qubits, smiles_list, entangling_layer = 'rxx', n_atom_to_qubit=1, interleaved='ryy', n_layers=1)
overlap_df_x_yy = quantum_overlap(num_qubits, smiles_list, entangling_layer = 'ryy', n_atom_to_qubit=1, interleaved='ryy', n_layers=1)
#TODO: test with different n_atom_to_qubit values
overlap_df_x_zz = quantum_overlap(num_qubits, smiles_list, entangling_layer = 'rzz', n_atom_to_qubit=1, interleaved='ryy', n_layers=1)
overlap_df = pd.DataFrame(overlap_df_x_zz, index=smiles_list, columns=smiles_list)
"""
overlap_df_rounded = overlap_df.round(6)
fig, ax = subplots(figsize=(12, 8))
ax.axis('off')
plt.title("QM similarity overlap matrix: benzene to pentacene ")
table = ax.table(cellText = overlap_df_rounded.values, rowLabels = overlap_df_rounded.index ,colLabels = overlap_df_rounded.columns, loc = 'center', cellLoc = 'center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig('QM_table_4.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("QM table 4 made")
"""
# %%
### Visualisation QMSE vs RDKit similarities ###

from matplotlib import pyplot as plt

x_low = 2 # "CC"
x_high = 10 # "CCCCCCCCCC"
x = np.arange(x_low,x_high+1,1)
fig, ax = plt.subplots()

ax.plot(x, tanimoto_matrix[0, :])
ax.plot(x, overlap_df_x_xx[0, :])
ax.plot(x, overlap_df_x_yy[0, :])
ax.plot(x, overlap_df_x_zz[0, :])
ax.set_xticks(x, minor=True)
ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
ax.set_title('QMSE overlaps', fontsize=16)
ax.set_xlabel('Number of carbon atoms of linear alkane $j$')
ax.set_ylabel('Fingerprint similarity')
ax.set_xlim(x_low, x_high)
ax.legend(['RDKit', '$R_x$ + $R_{xx}$', '$R_x$ + $R_{yy}$', '$R_x$ + $R_{zz}$'])
# ax.legend(['RDKit', '$R_x$ + CX + $R_{xx}$', '$R_x$ + CX + $R_{yy}$', '$R_x$ + CX + $R_{zz}$'])
ax.tick_params(axis='x', which='both', top=False)
ax.tick_params(axis='y', which='both', right=False)
plt.savefig("figure_aromatic3.pdf", format="pdf", dpi=2000)





"""
#%%
join_pdf("circuit.pdf", "Test1.pdf", "sphereplot.pdf", "tanimoto_table.pdf")
"""