# -*- coding: utf-8 -*-
'''
this code verifies the validation of qiskit qft and my own inverse qft, vice versa.
'''


import numpy as np
from sympy import true

np.fft.fft(np.array([1,0,0,1]))

from qiskit.circuit.library import QFT
from numpy import pi
# importing Qiskit
from qiskit.providers.aer import AerSimulator
sim = AerSimulator()  # make new simulator object
from qiskit import QuantumCircuit, transpile, Aer, IBMQ,assemble,execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt
def build_circuit(qc,n):
    for i in range(n):
        qc.cp(pi/2**(n-i), i, n)

def build_qft(arr,measure = 0):
    n = len(arr)
    if measure == 0:
        qc = QuantumCircuit(n)
    else:
        qc = QuantumCircuit(n,n)
    for i in range(n):
      if arr[i] == 1:
        qc.x(i)
    for i in range(n-1,-1,-1):
        qc.h(i)
        build_circuit(qc,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    if measure == 1:
        qc.measure(list(range(0,n)),list(range(0,n)))
    return qc
def main(target):
    for i in range(target-1,target):
        check(i)
def check(num):
    print(num)

    num = bin(num)
    arr = []
    for i in range(len(num)):
        arr.append(num[i])
    arr = arr[2:len(arr)]
    arr = list(map(int,arr))
    # print(arr)
    backend = Aer.get_backend('aer_simulator')
    # arr = [1,1,1,1,0]
    arr_str = ""
    for i in arr:
            arr_str+= str(i)
    # print("arr_str",arr_str)
    qft = build_qft(arr)
    qft_m = build_qft(arr,measure=1)
    job = execute(qft_m, backend)
    result = job.result()
    # print("qft input:",arr)
    print(result.get_counts())
    n = len(arr)

    bi_qft =QFT(n,inverse=True)
    test = QuantumCircuit(n,n)

    bi_qft = qft.compose(bi_qft)
    bi_qft = test.compose(bi_qft)
    
    bi_qft.measure(list(range(0,n)),list(range(0,n)))
    # bi_qft.draw(output='mpl')
    # plt.show()



    job = execute(bi_qft, backend)
    result = job.result()
    # print("after iqft input:",arr)
    print(result.get_counts())
    for k,j in result.get_counts().items():
        if(k==arr_str[::-1]):
            print("yes")
if __name__ == "__main__":
    main(100)