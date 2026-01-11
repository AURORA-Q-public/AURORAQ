import cirq
import qsimcirq
import numpy as np
import time
import sys

qubits = cirq.LineQubit.range(32)
np.random.seed(42)

gpu_options = qsimcirq.QSimOptions(
    use_gpu=True,
    gpu_mode=0,
    max_fused_gate_size=4
)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

def run_simulation(circuit, label=""):
    start = time.time()
    result = qsim_simulator.simulate(circuit)
    sys.stdout.flush()
    elapsed = time.time() - start
    print(f'{label} runtime: {elapsed:.6f} seconds.')
    return result

def vqc_circuit(qubits, layers=12):
    circuit = cirq.Circuit()
    n = len(qubits)
    for _ in range(layers):
        # 단일 큐비트 회전 게이트
        for q in qubits:
            circuit.append(cirq.rx(np.random.uniform(0, 2*np.pi))(q))
            circuit.append(cirq.ry(np.random.uniform(0, 2*np.pi))(q))
        # 인접 CZ 엔탱글먼트
        for i in range(n - 1):
            circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    return circuit

vqc_circ = vqc_circuit(qubits, layers=18)
result = run_simulation(vqc_circ, label="VQC")

# ✅ 최종 상태벡터 출력 (앞부분만 샘플링)
state_vector = result.final_state_vector
print("✅ Final state vector length:", len(state_vector))
print("✅ First 10 amplitudes (real parts):")
for i in range(10):
    print(f"  state[{i}]: {state_vector[i].real:.10f} + {state_vector[i].imag:.10f}j")

