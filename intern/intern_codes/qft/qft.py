import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

"""
this code build a module of qft
"""
def build_circuit(qc,n):
    for i in range(n):
        qc.cp(pi/2**(n-i), i, n)

def build_qft(n):
    qc = QuantumCircuit(n)
    for i in range(n-1,-1,-1):
        qc.h(i)
        build_circuit(qc,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    return qc

if __name__ == '__main__':
    print("qft:")
        qft = build_qft(1000)
        qft.draw(output='mpl')
        plt.show()