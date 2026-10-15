# %%
import sys

sys.path.append("../")
# %%
### QMSE scheme ###
import numpy as np
from rdkit import Chem
from quantum_molecular_encodings.supporting_functions import coulomb_matrix, matrix_to_circuit
from matplotlib import pyplot as plt
from PyPDF2 import PdfMerger 

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
circuit.draw(reverse_bits=True, output='mpl')
plt.savefig("circuit.pdf", format="pdf", bbox_inches='tight')
# %%
from qiskit.visualization import plot_state_qsphere
from qiskit.quantum_info import Statevector

statevector = np.array(Statevector(circuit))
plot_state_qsphere(statevector, figsize=(6, 6)).savefig('sphereplot.pdf')




#%%
join_pdf("circuit.pdf", "Test1.pdf", "sphereplot.pdf")