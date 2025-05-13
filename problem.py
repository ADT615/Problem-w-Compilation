from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
import numpy as np 
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
problem = driver.run()
hamiltonian = problem.hamiltonian

dipole: ElectronicDipoleMoment  = problem.properties.electronic_dipole_moment

if dipole is not None:
    nuclear_dip = dipole.nuclear_dipole_moment
    # Cập nhật moment lưỡng cực cho các thành phần x, y, z
    dipole.x_dipole.alpha += PolynomialTensor({"": nuclear_dip[0]})
    dipole.y_dipole.alpha += PolynomialTensor({"": nuclear_dip[1]})
    dipole.z_dipole.alpha += PolynomialTensor({"": nuclear_dip[2]})
    print("Đã thêm moment lưỡng cực vào Hamiltonian.")
else:
    print("Moment lưỡng cực không tồn tại trong problem.")

# Kiểm tra coefficients của Hamiltonian
coefficients = hamiltonian.electronic_integrals

second_q_op = hamiltonian.second_q_op()

mapper = JordanWignerMapper()
qubit_p_op = mapper.map(second_q_op) # H_0 dưới dạng SparsePauliOp

# Chuyển toán tử sang dạng ma trận

Hopt = qubit_p_op # Hamiltonian tĩnh (static)
H_static = Hopt.to_matrix() # H_static dưới dạng ma trận

mapper = JordanWignerMapper()

dipole_ops = dipole.second_q_ops()
#print("Nội dung của dipole_ops:", dipole_ops)
# Lấy toán tử moment lưỡng cực từ phương Z (vì X và Y rỗng)
dipole_op = dipole_ops["ZDipole"]
dipole_qubit = mapper.map(dipole_op) # Dipole dưới dạng SparsePauliOp
dipole_matrix = dipole_qubit.to_matrix()

# Hàm tính Hamiltonian H(t) dạng SparsePauliOp
def Hamilton1(t, Hopt, dipole_qubit, E0, Gamma):
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t**2)
    V_t = f_t*dipole_qubit
    H_total1 = Hopt + V_t
    return H_total1

import qiskit

from scipy.linalg import expm
def time_dependent(num_qubits : int , H_total_t,t):
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for qubit in range(num_qubits//2, num_qubits):
        qc.x(qubit)

    if not np.allclose(H_total_t(t).to_matrix(), np.conj(H_total_t(t).to_matrix()).T):    
        raise ValueError("The Hamiltonian is not Hermitian.")    
    
    time_points = np.linspace(0,t,100)

    # Calculate the integral of H(t) using numerical approximation (e.g., trapezoidal rule)
    integral = np.zeros_like(H_total_t, dtype=complex)

    for i in range(len(time_points) - 1):
        dt = time_points[i+1] - time_points[i]
        integral += (H_total_t(time_points[i]) + H_total_t(time_points[i + 1])) / 2 * dt

   # Compute the matrix exponential
    U = expm(-1j * integral)  

      # Check if U is unitary
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("The resulting matrix U is not unitary.")
        
    #return U matrix
    """
    # Create a UnitaryGate from the unitary_matrix
    unitary_gate = UnitaryGate(U)

    # Append the unitary_gate to the quantum circuit
    qc.append(unitary_gate, range(qc.num_qubits))
    """
    from qoop.core.state import specific_matrix
    
    return specific_matrix(U)
    

def time_dependent_integral(H_total_t, t):
    """create U circuit from h_opt and time t
    
    Args:
        - qc (QuantumCircuit): Init circuit
        - h_opt: Hamiltonian
        - t (float): time
        
    Returns:
        - QuantumCircuit: the added circuit
    """
    # Ensure h_opt is Hermitian
    if not np.allclose((H_total_t(t).to_matrix()), np.conj(H_total_t(t).to_matrix()).T):
        raise ValueError("The Hamiltonian is not Hermitian.")

    time_points = np.linspace(0, t, 100)
    # Calculate the integral of H(t) using numerical approximation (e.g., trapezoidal rule)
    integral = np.zeros_like(H_total_t, dtype=complex)  # Initialize integral as a matrix
    
    for i in range(len(time_points) - 1):
        dt = time_points[i + 1] - time_points[i]
        integral += (H_total_t(time_points[i]) + H_total_t(time_points[i + 1])) / 2 * dt

    return integral    

num_qubits = Hopt.num_qubits

def H_time(t):
    Gamma = 0.25
    E0 = 0.01
    return Hamilton1(t, Hopt, dipole_qubit, E0, Gamma)

def trotter_circuit(nqubits, labels, coeffs, t, M):

    # Convert one Trotter decomposition ,e^{iZ_1Z_2*delta}*e^{iZ_2Z_3*delta}*...e^{iZ_nZ_1*delta} to a quantum gate
    circuit = qiskit.QuantumCircuit(nqubits)
    for qubit in range(nqubits//2, nqubits):
        circuit.x(qubit)

    # Time increment range
    delta = 0.5
        
    for i in range(len(labels)):
        # 'IX', 'IZ', 'IY' case
        if labels[i][0] == 'I':
            if labels[i][1] == 'Z':
                circuit.rz(2*delta*coeffs[i],1)
            elif labels[i][1] == 'X':
                circuit.rx(2*delta*coeffs[i],1)
            elif labels[i][1] == 'Y':
                circuit.ry(2*delta*coeffs[i],1)
    
        # 'XI', 'ZI', 'YI' case
        elif labels[i][1] == 'I':
            if labels[i][0] == 'Z':
                circuit.rz(2*delta*coeffs[i],0)
            elif labels[i][0] == 'X':
                circuit.rx(2*delta*coeffs[i],0)
            elif labels[i][0] == 'Y':
                circuit.ry(2*delta*coeffs[i],0)
    
        # # 'XX', 'ZZ', 'YY' case
        elif labels[i] in ['XX', 'YY', 'ZZ']:
            for j in range(nqubits):
                if labels[i][1] == 'Z':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rzz(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'X':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rxx(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'Y':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.ryy(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
    return circuit


from qoop.compilation.qsp import QuantumStatePreparation
import sys
import numpy as np
import qiskit
def H_time(t):
    Gamma = 0.25
    E0 = 0.01
    return Hamilton1(t, Hopt, dipole_qubit, E0, Gamma)

p0s = []
N = 4
T = 10
labels = time_dependent_integral(H_time,t=T).paulis.to_labels()
coeffs = time_dependent_integral(H_time,t=T).coeffs
coeffs = np.real(coeffs)
times = np.linspace(0,10,4)
qc = trotter_circuit(N,labels, coeffs, T, M=100)
for time in times:
    qsp = QuantumStatePreparation(
        u=qc,
        target_state= time_dependent(num_qubits,H_time,time).inverse()
        ).fit(num_steps=30, metrics_func=['loss_basic'])
    p0s.append(1-qsp.compiler.metrics['loss_basic'][-1])

print('Mean loss',p0s)



