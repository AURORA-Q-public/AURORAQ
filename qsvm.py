import cirq
import qsimcirq
import numpy as np
import time
import sys

# ✅ 1. 큐비트 설정 (28 큐비트로 변경)
qubits = cirq.LineQubit.range(32)

# ✅ 2. QSim 시뮬레이터 옵션
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

# ✅ 4. QSVM 회로 (기존 구조 유지)
def qsvm_circuit(qubits, data_vector):
    circuit = cirq.Circuit()
    # Feature map (angle embedding)
    for i, q in enumerate(qubits):
        if i < len(data_vector):
            circuit.append(cirq.ry(data_vector[i])(q))
    # Simple entanglement
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    return circuit

np.random.seed(42)   # ✅ QSVM도 이 한 줄 필요함
data = np.random.uniform(0, 2*np.pi, size=len(qubits))


# ✅ 6. 회로 생성 및 실행
qsvm_circ = qsvm_circuit(qubits, data)
result = run_simulation(qsvm_circ, label="QSVM")

# ✅ 7. 최종 상태벡터 일부 출력
state_vector = result.final_state_vector
print("✅ Final state vector length:", len(state_vector))
print("✅ First 10 amplitudes:")
for i in range(10):
    print(f"  state[{i}]: {state_vector[i].real:.10f} + {state_vector[i].imag:.10f}j")
