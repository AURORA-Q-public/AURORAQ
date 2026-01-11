import cirq
import qsimcirq
import numpy as np
import time
import sys

# ✅ 0. 랜덤 시드 고정 (정합성 검증 및 재현성 확보용)
np.random.seed(42)

# -----------------------
# 1. 큐비트 설정 (28 큐비트)
# -----------------------
n_qubits = 32
qubits = cirq.LineQubit.range(n_qubits)

# -----------------------
# 2. 깊은 VQE 스타일 ansatz 생성
# -----------------------
def vqe_ansatz(qubits, layers=30):  # ✅ 반복 횟수(layers) 조절해서 depth 제어
    circuit = cirq.Circuit()
    n = len(qubits)

    for _ in range(layers):
        # 단일 큐비트 회전 계층
        for q in qubits:
            theta = np.random.rand() * 2 * np.pi
            phi = np.random.rand() * 2 * np.pi
            circuit.append(cirq.ry(theta)(q))
            circuit.append(cirq.rz(phi)(q))

        # 엔탱글먼트 계층 (순차 CNOT 체인)
        for i in range(n - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return circuit

# -----------------------
# 3. 회로 생성 (예: 30 Layer → 깊이 수백~수천 수준)
# -----------------------
vqe_circuit = vqe_ansatz(qubits, layers=30)
print("✅ 생성된 회로 깊이 (게이트 수):", len(vqe_circuit))

# -----------------------
# 4. GPU 시뮬레이터 실행
# -----------------------
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0)
simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

start = time.time()
result = simulator.simulate(vqe_circuit, initial_state=0)
end = time.time()

print("✅ 시뮬레이션 완료")
print("⏱️ Runtime:", end - start, "초")

# -----------------------
# 5. 최종 상태벡터 출력
# -----------------------
state_vector = result.final_state_vector
print("✅ Final state vector length:", len(state_vector))
print("✅ First 10 amplitudes:")
for i in range(10):
    print(f"  state[{i}]: {state_vector[i].real:.10f} + {state_vector[i].imag:.10f}j")
