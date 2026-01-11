# AURORA-Q
**AURORA-Q: Asynchronous Unified Resource Optimizer for Quantum Simulation on HPC System** (Submitted ICDCS'26) 

## Overview
AURORA-Q is a tiered-memory quantum circuit simulation framework designed to overcome the memory bottleneck and scalability limits of large-scale quantum circuit simulation with leadership-scale HPC systems. 
To efficiently support simulations that exceed physical GPU VRAM capacity, AURORA-Q introduces:
- **A unified tiered-memory architecture** that integrates GPU VRAM, DRAM, and storage (Lustre PFS) into a single simulation framework.
- **An asynchronous execution pipeline** that overlaps computation with data movement to hide high I/O latency.
- **A dynamic two-level cache hierarchy** (LFU for GPU, Score-based for DRAM) with adaptive resource control to exploit access locality.

The framework is built upon Google’s **Qsim**, but redesigned to support a tiered-memory hierarchy, modifying approximately 1,500 lines of the core engine. 
Our evaluation demonstrates that AURORA-Q executes up to **43-qubit simulations on 512 GPUs**, a scale unattainable by existing state-of-the-art simulators due to limited memory resources. AURORA-Q achieves up to **2.91× speedup** over baselines while maintaining a **79.3% cache hit rate**.

## Key Features
- **Scalable to 43 Qubits**: Breaks the "memory wall" by utilizing the entire memory hierarchy.
- **Decoupled Residency Management**: Separates global logical ownership from local physical residency, eliminating global synchronization overhead.
- **Asynchronous I/O-Compute Overlap**: Uses multiple CUDA streams to mask storage/DRAM fetch latency with gate execution.
- **Adaptive Resource Control**: Dynamically adjusts cache sizes and kernel parameters (block/thread counts) to prevent OOM and maximize throughput.


## Modified Components
AURORA-Q extends and replaces the static memory model of Qsim with the following core modules:
- **Tiered Memory State Layout (III-A)**: Logical state partitioning and on-demand subset generation.
- **Async Execution Pipeline (III-B)**: Per-task coordination via Ownership Map and overlapped execution.
- **Two-Level Cache Hierarchy (III-C)**: LFU-based GPU cache controller and score-based DRAM cache controller.
- **Adaptive Resource Control (III-D)**: Context-aware runtime control for GPU assignment and kernel parameter tuning.

## Modified Modules
AURORA-Q modifies and extends the following core Qsim modules:
- `simulator_cuda.h`  
- `vectorspace_cuda.h`  
- `simulator_cuda_kernel.h`
- `pybind_cuda.cpp`
  
