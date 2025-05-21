from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
import numpy as np 
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA , SLSQP, L_BFGS_B, SPSA, NELDER_MEAD
from qiskit_algorithms import VQE
from qoop.compilation.qsp import QuantumStatePreparation
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit_algorithms.eigensolvers import NumPyEigensolver

from scipy.linalg import expm
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

ansatz = UCC(
    num_spatial_orbitals = problem.num_spatial_orbitals, #2
    num_particles = problem.num_particles, # [1, 1]
    excitations='sd',
    qubit_mapper=mapper,
    initial_state=HartreeFock(
        num_spatial_orbitals = problem.num_spatial_orbitals,
        num_particles = problem.num_particles, 
        qubit_mapper=mapper,
    ),
    reps=1,

)

#from qiskit.utils import QuantumInstance
estimator = Estimator()
optimizer = SLSQP(maxiter=200)
vqe = VQE(estimator, ansatz, optimizer)
res = vqe.compute_minimum_eigenvalue(qubit_p_op)

numpy_solver = NumPyEigensolver()
exact_result = numpy_solver.compute_eigenvalues(qubit_p_op)
ref_value = exact_result.eigenvalues
#print(f"Reference value: {ref_value  }")
#print(f"VQE values: {res.optimal_value }")
#print(f"Delta from reference energy value is {(res.optimal_value - ref_value)}")

Hopt = qubit_p_op # Hamiltonian tĩnh (static)
H_static = Hopt.to_matrix() # H_static dưới dạng ma trận

mapper = JordanWignerMapper()

dipole_ops = dipole.second_q_ops()
#print("Nội dung của dipole_ops:", dipole_ops)
# Lấy toán tử moment lưỡng cực từ phương Z (vì X và Y rỗng)
dipole_op = dipole_ops["ZDipole"]
dipole_qubit = mapper.map(dipole_op) # Dipole dưới dạng SparsePauliOp
dipole_matrix = dipole_qubit.to_matrix()


def Hamilton(t, H_static, dipole_matrix, E0, Gamma):
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t**2)
    V_t = f_t*dipole_matrix
    H_total = H_static + V_t
    return H_total

def Hamilton1(t, Hopt, dipole_qubit, E0, Gamma):
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t**2)
    V_t = f_t*dipole_qubit
    H_total1 = Hopt + V_t
    return H_total1

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

def H_time(t):
    Gamma = 0.25
    E0 = 0.01
    return Hamilton1(t, Hopt, dipole_qubit, E0, Gamma)

def create_parameterized_hamiltonian_ansatz(nqubits, pauli_labels_fixed, num_layers, initial_coeffs_values=None, add_classical_bits=True):
    """
    Tạo một Hamiltonian ansatz có tham số U(theta) = product_L (product_k exp(-i * theta_Lk * P_k)).

    Args:
        nqubits (int): Số qubit.
        pauli_labels_fixed (list[str]): Danh sách các chuỗi Pauli CỐ ĐỊNH (P_k).
        num_layers (int): Số lớp (N_L) của ansatz.
        initial_coeffs_values (list[float] or np.ndarray, optional): 
                               Giá trị khởi tạo cho các tham số theta. 
                               Nếu num_layers > 1, đây có thể là một mảng 1D dẹt (flattened)
                               hoặc có thể cấu trúc nó theo lớp. 
                               Độ dài phải là len(pauli_labels_fixed) * num_layers.
        add_classical_bits (bool): Nếu True, thêm bit cổ điển.

    Returns:
        qiskit.QuantumCircuit: Mạch lượng tử ansatz có tham số.
        list[Parameter]: Danh sách tất cả các tham số đã tạo (dạng dẹt).
    """
    if not pauli_labels_fixed:
        raise ValueError("pauli_labels_fixed không được rỗng.")

    if add_classical_bits:
        circuit = QuantumCircuit(nqubits, nqubits)
    else:
        circuit = QuantumCircuit(nqubits)

    total_parameters = len(pauli_labels_fixed) * num_layers # Tạo tham số $ \theta_{Lk}$
    params_vector = ParameterVector('θ', total_parameters)
    
    all_parameters_list = list(params_vector) # Danh sách các đối tượng Parameter

    param_idx_counter = 0
    for _ in range(num_layers): # Lặp qua các lớp
        for pauli_str in pauli_labels_fixed: # Lặp qua các toán tử Pauli cố định
            if not pauli_str or len(pauli_str) != nqubits:
                raise ValueError(f"Chuỗi Pauli '{pauli_str}' không hợp lệ.")
            
            current_pauli_operator = Pauli(pauli_str)
            # time trong PauliEvolutionGate bây giờ là một Parameter
            param_coeff = params_vector[param_idx_counter]
            param_idx_counter += 1
            
            evolution_gate = PauliEvolutionGate(current_pauli_operator, time=param_coeff) 
            circuit.append(evolution_gate, range(nqubits))
        if num_layers > 1 : # Thêm barrier giữa các lớp nếu có nhiều lớp
            circuit.barrier()

    return circuit, all_parameters_list

# Lấy labels từ Hopt (phần tĩnh)
static_pauli_op = Hopt 
static_labels = static_pauli_op.paulis.to_labels()

# Lấy labels từ dipole_qubit (phần tương tác)
dipole_pauli_op = dipole_qubit
dipole_interaction_labels = dipole_pauli_op.paulis.to_labels()

combined_unique_labels = list(dict.fromkeys(static_labels + dipole_interaction_labels))
num_qubits = Hopt.num_qubits

# 2. Tạo Ansatz U(theta)
N = num_qubits # num_qubits = 4
num_ansatz_layers = 1 # Số lớp cho ansatz, thử nghiệm

ansatz_u, optimizable_parameters = create_parameterized_hamiltonian_ansatz(
    nqubits=N,
    pauli_labels_fixed=combined_unique_labels,
    num_layers=num_ansatz_layers,
    add_classical_bits=True # Giữ True để nhất quán với target_state nếu nó có clbits
    # initial_coeffs_values có thể được truyền vào optimizer sau
)

print("Ansatz U(theta) được tạo:")
print(ansatz_u.draw(output='text'))
print(f"Số lượng tham số có thể tối ưu: {len(optimizable_parameters)}")
print(f"Các tham số: {optimizable_parameters}")

circuit = ansatz.assign_parameters(res.optimal_parameters)
psi_0_vqe = np.array(Statevector(circuit).data)

p0s = []
all_psi_t_approximated = {} 

times_for_qsp = np.linspace(0, 10, 4) 

for t_current in times_for_qsp: 
    # 1. Tạo target_state V(t_current)^dagger cho QSP tại thời điểm t_current này
    target_unitary_at_t_circuit = time_dependent(num_qubits, H_time, t_current)

    # 2. Tạo initial_point cho tham số theta dựa trên t_current
    current_H_int = time_dependent_integral(H_time, t=t_current) # $c_k(t) từ H(t)$
    initial_point_for_this_t = [] # $\vec{\theta_0}$
    label_to_coeff_map_current_t = dict(zip(current_H_int.paulis.to_labels(), current_H_int.coeffs.real))
    for _ in range(num_ansatz_layers): # Nếu có nhiều lớp, lặp lại bộ giá trị khởi tạo
        for label in combined_unique_labels:
            initial_point_for_this_t.append(label_to_coeff_map_current_t.get(label, np.random.rand())) # Lấy coeff hoặc ngẫu nhiên 

    qsp_instance = QuantumStatePreparation(
        u=ansatz_u.copy(),
        target_state=target_unitary_at_t_circuit.inverse(),
    ).fit(
        num_steps=300, 
        metrics_func=['loss_basic'],
        initial_point=np.array(initial_point_for_this_t) 
    )
    
    # 4. Lấy các tham số tối ưu theta_t_current* cho thời điểm t_current này
    print(f"t = {t_current}, optimal_thetas = {qsp_instance.thetas}")
    
    # 5. Tạo mạch ansatz U(theta_t_current*) 
    U_theta_t_current_optimized = ansatz_u.assign_parameters(qsp_instance.thetas)
    
    # 6. TÍNH |Psi(t_current)> XẤP XỈ = U(theta_t_current*) |Psi_0_vqe>
    U_theta_t_current_matrix = Operator(U_theta_t_current_optimized).data
    psi_0_vqe_col = psi_0_vqe.reshape(-1, 1)
    psi_t_current_col_approx = U_theta_t_current_matrix @ psi_0_vqe_col
    psi_t_current_approx = psi_t_current_col_approx.flatten()
    
    all_psi_t_approximated[t_current] = psi_t_current_approx # Lưu lại

    if 'loss_basic' in qsp_instance.compiler.metrics:
        p0s.append(1 - qsp_instance.compiler.metrics['loss_basic'][-1])
    else:
        p0s.append(0)
    print("Fidelity vs ansatz co tham so:", p0s)

t_target_print = times_for_qsp[0]                          
print(f"\n--- In vector trạng thái Psi(t) tại t = {t_target_print:.4f} ---")

if t_target_print in all_psi_t_approximated:
    psi_at_t_target = all_psi_t_approximated[t_target_print]
    if psi_at_t_target is not None:
        print("Vector trạng thái Psi(t) xấp xỉ:")
        print(psi_at_t_target)
        print(f"Chuẩn của Psi(t={t_target_print:.4f}): {np.linalg.norm(psi_at_t_target)}")        
    else:
        print(f"Không có dữ liệu Psi(t) hợp lệ cho t = {t_target_print:.4f} (do loss cao hoặc lỗi).")
else:
    closest_t = min(all_psi_t_approximated.keys(), key=lambda k: abs(k - t_target_print) if all_psi_t_approximated[k] is not None else float('inf'))
    if all_psi_t_approximated.get(closest_t) is not None and abs(closest_t - t_target_print) < (times_for_qsp[1]-times_for_qsp[0])/2 : # Ngưỡng để coi là "gần"
        print(f"Không tìm thấy chính xác t = {t_target_print:.4f}. Hiển thị cho thời điểm gần nhất t = {closest_t:.4f}:")
        psi_at_closest_t = all_psi_t_approximated[closest_t]
        print(psi_at_closest_t)
        print(f"Chuẩn của Psi(t={closest_t:.4f}): {np.linalg.norm(psi_at_closest_t)}")
    else:
        print(f"Không tìm thấy dữ liệu Psi(t) cho t = {t_target_print:.4f} hoặc các điểm lân cận.")











