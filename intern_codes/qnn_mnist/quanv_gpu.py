# -*- coding: utf-8 -*-
"""
This code use quantum layer as convulotion 2d layers and use CPU to train the model.
However, i never train the model with this code since my computer's CPU is not good enough.
"""
## 1. Module Import, Select Device, and Download MNIST Data
"""

# !pip install qiskit

"""### 1-1. Setup & Module Import"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary

import qiskit
from qiskit.visualization import *
from qiskit.circuit.random import random_circuit

from itertools import combinations

"""### 1-2. Select Device"""

if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

"""### 1-3. Setting of the main hyper-parameters of the model"""

BATCH_SIZE = 1
EPOCHS = 1     # Number of optimization epochs
n_layers = 1    # Number of random layers
n_train = 10   # Size of the train dataset
n_test = 10    # Size of the test dataset

SAVE_PATH = "quanvolution/" # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
seed = 47
np.random.seed(seed)        # Seed for NumPy random number generator
torch.manual_seed(seed)     # Seed for TensorFlow random number generator

"""### 1-4. Data Loading"""

train_dataset = datasets.MNIST(root = "./data",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "./data",
                              train = False,
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(1):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))

"""## 2. Construct Quantum Circuit"""

from qiskit import transpile,assemble

"""### 2-1. Create a 'Quantum Class' with Qiskit"""

class QuanvCircuit:
    """ 
    This class defines filter circuit of Quanvolution layer
    """
    
    def __init__(self, kernel_size, backend, shots, threshold):
        # --- Circuit definition start ---
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter('theta{}'.format(i)) for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        
        self._circuit.barrier()
        self._circuit.compose( random_circuit(self.n_qubits, 2))
        self._circuit.measure_all()
        # ---- Circuit definition end ----

        self.backend   = backend
        #self.backend.set_options(device='GPU')
        self.shots     = shots
        self.threshold = threshold

    def run(self, data):
        # data shape: tensor (1, 5, 5)
        # val > self.threshold  : |1> - rx(pi)
        # val <= self.threshold : |0> - rx(0)

        # reshape input data
        # [1, kernel_size, kernel_size] -> [1, self.n_qubits]
        data = torch.reshape(data, (1, self.n_qubits))

        # encoding data to parameters
        thetas = []
        self.threshold =0 
        for dat in data:
            theta = []
            for val in dat:
                #print(val)
                if val > self.threshold:
                    theta.append(np.pi)
                else:
                    theta.append(0)
            thetas.append(theta)
        
        param_dict = dict()
        for theta in thetas:
            for i in range(self.n_qubits):
                param_dict[self.theta[i]] = theta[i]
        param_binds = [param_dict]
        t_qc = transpile(self._circuit,
                         self.backend)
        #print(t_qc)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds =[param_dict] )
        
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # execute random quantum circuit
                
        """
        job = qiskit.execute(self._circuit, 
                             self.backend, 
                             shots = self.shots, 
                             parameter_binds = param_binds)
        result = job.result().get_counts()
        
        """
        # decoding the result
        #print(result)
        counts = 0
        for key, val in result.items():
            cnt = sum([int(char) for char in key])
            counts += cnt * val

        # Compute probabilities for each state
        probabilities = counts / (self.shots * self.n_qubits)
        # probabilities = counts / self.shots
        
        return probabilities

"""Let's test the implementation."""

backend = qiskit.Aer.get_backend('qasm_simulator')

filter_size = 2
circ = QuanvCircuit(filter_size, backend, 100, 127)
data = torch.tensor([[0, 200], [100, 255]])

print(data.size())
print(circ.run(data))

circ._circuit.draw()

"""### 2-2. Create a 'Quanvolution Class' with PyTorch"""

# def quanv_feed(image):
#     """
#     Convolves the input image with many applications 
#     of the same quantum circuit.

#     In the standard language of CNN, this would correspond to
#     a convolution with a 5×5 kernel and a stride equal to 1.
#     """
#     out = np.zeros((24, 24, 25))

#     # Loop over the coordinates of the top-left pixel of 5X5 squares
#     for j in range(24):
#         for k in range(24):
#             # Process a squared 5x5 region of the image with a quantum circuit
#             circuit_input = []
#             for a in range(5):
#                 for b in range(5):
#                     circuit_input.append(image[j + a, k + b, 0])
#             q_results = circuit(circuit_input)

#             # Assign expectation values to different channels of the output pixel (j/2, k/2)
#             for c in range(25):
#                 out[24, 24, c] = q_results[c]
#     return out

class QuanvFunction(Function):
    """ Quanv function definition """
    
    @staticmethod
    def forward(ctx, inputs, in_channels, out_channels, kernel_size, quantum_circuits, shift):
        """ Forward pass computation """
        # input  shape : (-1, 1, 28, 28)
        # otuput shape : (-1, 6, 24, 24)
        ctx.in_channels      = in_channels
        ctx.out_channels     = out_channels
        ctx.kernel_size      = kernel_size
        ctx.quantum_circuits = quantum_circuits
        ctx.shift            = shift

        _, _, len_x, len_y = inputs.size()
        len_x = len_x - kernel_size + 1
        len_y = len_y - kernel_size + 1
        
        features = []
        for input in inputs:
            feature = []
            for circuit in quantum_circuits:
                xys = []
                for x in range(len_x):
                    ys = []
                    for y in range(len_y):
                        data = input[0, x:x+kernel_size, y:y+kernel_size]
                        ys.append(circuit.run(data))
                    xys.append(ys)
                feature.append(xys)
            features.append(feature)       
        result = torch.tensor(features)

        ctx.save_for_backward(inputs, result)
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Quanv(nn.Module):
    """ Quanvolution(Quantum convolution) layer definition """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 backend=qiskit.Aer.get_backend('qasm_simulator'), 
                 shots=100, shift=np.pi/2):
        super(Quanv, self).__init__()
        self.quantum_circuits = [QuanvCircuit(kernel_size=kernel_size, 
                                              backend=backend, shots=shots, threshold=0) 
                                 for i in range(out_channels)]
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.shift        = shift
        
    def forward(self, inputs):
        return QuanvFunction.apply(inputs, self.in_channels, self.out_channels, self.kernel_size,
                                   self.quantum_circuits, self.shift)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quanv = Quanv(1, 6, kernel_size=2)
        self.conv = nn.Conv2d(6, 10, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.quanv(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim = 10)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

epochs = 1
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculating loss
        loss = loss_func(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))

model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data).cuda()
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100 / 2)
        )