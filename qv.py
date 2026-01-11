import cirq
import qsimcirq
import numpy as np
import time
import sys

# ✅ 1. 큐비트 28개로 변경
qubits = cirq.LineQubit.range(32)
np.random.seed(42)
# ✅ 2. QSim 시뮬레이터 설정
gpu_options = qsimcirq.QSimOptions(
    use_gpu=True,
    gpu_mode=0,
    max_fused_gate_size=4
)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

# ✅ 3. 시뮬레이션 함수
def run_simulation(circuit, label=""):
    start = time.time()
    result = qsim_simulator.simulate(circuit)
    sys.stdout.flush()
    elapsed = time.time() - start
    print(f'{label} runtime: {elapsed:.6f} seconds.')
    return result

# ✅ 4. QV 회로 생성 (원래 구조 그대로 유지)
def qv_circuit(qubits, depth=15):
    circuit = cirq.Circuit()
    n = len(qubits)
    for d in range(depth):
        np.random.shuffle(qubits)
        for i in range(0, n - 1, 2):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            # U 게이트 구성 (Rz * Ry * Rz)
            u = cirq.unitary(cirq.rz(lam)) @ cirq.unitary(cirq.ry(theta)) @ cirq.unitary(cirq.rz(phi))
            u_gate = cirq.MatrixGate(u)
            circuit.append(u_gate(qubits[i]))
            circuit.append(cirq.CX(qubits[i], qubits[i+1]))
    return circuit

# ✅ 5. 회로 생성 및 시뮬레이션 실행
qv_circ = qv_circuit(qubits, depth=15)
result = run_simulation(qv_circ, label="QV")

# ✅ 6. 최종 상태벡터 일부 출력
state_vector = result.final_state_vector
print("✅ Final state vector length:", len(state_vector))
print("✅ First 10 amplitudes (real parts):")
for i in range(10):
    print(f"  state[{i}]: {state_vector[i].real:.10f} + {state_vector[i].imag:.10f}j")
