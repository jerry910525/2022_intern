from genericpath import samefile
from locale import normalize
from more_itertools import sample
import qiskit
from qiskit.providers.aer.extensions.snapshot_statevector import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

'''
this code verifies the validation of qiskit qft and fft in numpy.
'''

sample = []
num = 20
num = 2**num #since we can only have qubit number with the exponential of 2.
t = 0
for i in range(num):
    tmp = random.random()
    sample.append(tmp)
    t += tmp**2
# print(t)
# print("sample:",sample)

sample_norm = sample
for i in range(len(sample)):
    sample_norm[i] = sample[i] * (1/t**(1/2))

t = 0
for i in range(len(sample_norm)):
    
    t += sample_norm[i]**2
# print("sample_norm:",sample_norm)


n = len(sample_norm)
q = int(math.log(n,2))

"""
Start the QFT circuit
"""
circuit = qiskit.QuantumCircuit(q)
circuit.initialize(sample_norm, range(q))
circuit.snapshot_statevector('init') # This vector is correct
circuit += qiskit.circuit.library.QFT(num_qubits=q, do_swaps=True, approximation_degree=0)
circuit.snapshot_statevector('qft')
circuit.measure_all()
starttime = time.time()
# print(time.time()-starttime)
aer_sim = qiskit.Aer.get_backend('qasm_simulator')
qobj = qiskit.transpile(circuit, aer_sim)
result = aer_sim.run(qobj, shots=1).result()
init_vec = result.data()["snapshots"]["statevector"]["init"][0]
qft_vec = result.data()["snapshots"]["statevector"]["qft"][0]

print(time.time()-starttime)

"""
Start the FFT
"""
PSD_q = np.real(qft_vec * np.conj(qft_vec))
# print("qft:",PSD_q)
# Classical np.fft
starttime = time.time()
f = sample_norm
fhat = np.fft.fft(f,n)

PSD = np.real(fhat * np.conj(fhat) / n)
print(time.time()-starttime)

"""
plot the result
"""
# print("fft",PSD)
# plt.plot(PSD_q, color='r', linewidth=2, label='QTF')
# plt.plot(PSD, color='c', linestyle='dashed', linewidth=2, label='npFFT')
# plt.xlim(0, n)
# plt.legend()
# plt.show()

"""
result: the validation of qft in qiskit and fft in numpy is correct.
"""