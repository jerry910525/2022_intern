from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms import Shor

import numpy as np

from qiskit import QuantumCircuit,Aer,execute
from qiskit.tools.visualization import plot_histogram

"""
this code use the qiskit built-in function to implement shor algorithm.
"""

bkend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(bkend,shots = 1000)
shor = Shor(quantum_instance=quantum_instance)
result = shor.factor(N=15, a=2)
print('Factors:', result.factors)