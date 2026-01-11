// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#ifndef VECTORSPACE_CUDA_H_
#define VECTORSPACE_CUDA_H_

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //KCJ
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

#include <memory>
#include <utility>
#include <vector>
#include </opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include/mpi.h>
#include <chrono> 

namespace qsim {

namespace detail {

inline void do_not_free(void*) {}

inline void free(void* ptr) {
  //cudaFree(ptr);
}

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename FP>
class VectorSpaceCUDA {
 public:
  using fp_type = FP;
  static double total_cuda_malloc_time;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;
  static std::vector<fp_type*> multi_gpu_ptrs_;


 public:
  class Vector {
   public:
    Vector() : ptr_(nullptr, &detail::free), num_qubits_(0) {}

    Vector(Pointer&& ptr, unsigned num_qubits)
        : ptr_(std::move(ptr)), num_qubits_(num_qubits) {}

    Vector(const Vector& other)
        : ptr_(nullptr, &detail::free), num_qubits_(other.num_qubits_) {
      if (other.ptr_) {
        fp_type* p;
        //auto size = sizeof(fp_type) * Impl::MinSize(other.num_qubits_);
        auto size = sizeof(fp_type) * 64;
        cudaMalloc(&p, size);
        cudaMemcpy(p, other.ptr_.get(), size, cudaMemcpyDeviceToDevice);
        ptr_.reset(p);
      }
    }

    Vector& operator=(const Vector& other) {
      if (this != &other) {
        num_qubits_ = other.num_qubits_;
        if (other.ptr_) {
          fp_type* p;
          //auto size = sizeof(fp_type) * Impl::MinSize(other.num_qubits_);
          auto size = sizeof(fp_type) * 64;
          cudaMalloc(&p, size);
          cudaMemcpy(p, other.ptr_.get(), size, cudaMemcpyDeviceToDevice);
          ptr_.reset(p);
        } else {
          ptr_.reset(nullptr);
        }
      }
      return *this;
    }

    void set(fp_type* ptr, unsigned num_qubits) {
      ptr_.reset(ptr);
      num_qubits_ = num_qubits;
    }

    fp_type* get() {
  //    printf("KCJ DEBUG: Vector.get() called. Returning address: %p\n", ptr_.get());
      return ptr_.get();
    }

    const fp_type* get() const {
   //   printf("KCJ DEBUG: Vector.get() const ver. called. Returning address: %p\n", ptr_.get());
      return ptr_.get();
    }


    fp_type* release() {
      num_qubits_ = 0;
      return ptr_.release();
    }

    unsigned num_qubits() const {
      return num_qubits_;
    }

   bool requires_copy_to_host() const {
        return true;  // 항상 true를 반환
      }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
  };

  static std::vector<fp_type*>& MultiGPUPointers() {
    return multi_gpu_ptrs_;
  }

  static void ClearMultiGPUPointers() {
    for (auto ptr : multi_gpu_ptrs_) {
      cudaFree(ptr);
    }
    multi_gpu_ptrs_.clear();
  }

  static std::vector<Vector>& GlobalStateParts() {
    static std::vector<Vector> global_state_parts;
    return global_state_parts;
  }

  static Vector Create(unsigned num_qubits) {
    if (num_qubits == 0) {
      return Null();
    }

    if (!GlobalStateParts().empty() && GlobalStateParts()[0].num_qubits() == num_qubits) {
      return GlobalStateParts()[0];
    }

    fp_type* p;
    auto size = sizeof(fp_type) * 64;
    auto rc = cudaMalloc(&p, size);

    if (rc == cudaSuccess) {
      int num_gpus = 0;
      cudaError_t err = cudaGetDeviceCount(&num_gpus);

      if (err != cudaSuccess || num_gpus == 0) {
        num_gpus = 1;
      }

      GlobalStateParts() = CreateMultiGPU(num_qubits, num_gpus);
      cudaFree(p);
      //cudaDeviceSynchronize();
      return GlobalStateParts()[0];
    } else {
      return Null();
    }
  }



static std::vector<Vector> CreateMultiGPU(unsigned num_qubits, int num_gpus) {
    if (num_qubits == 0) {
        return GlobalStateParts();
    }

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    int node_id, total_nodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);

    if (!GlobalStateParts().empty()) {
        return GlobalStateParts();
    }

    ClearMultiGPUPointers();
    GlobalStateParts().clear();

    size_t total_size = MinSize(num_qubits);
    size_t size_per_node = (total_nodes > 1) ? total_size / total_nodes : total_size;
    size_t size_per_gpu  = size_per_node / num_gpus;

    // 🔧 Dummy 메모리 크기 (단일 vector slot만 할당)
    size_t Real_size_per_gpu = 64;

    std::vector<Vector> state_parts;

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        fp_type* gpu_ptr = nullptr;

        auto start_time = std::chrono::high_resolution_clock::now();

        cudaError_t err_before = cudaGetLastError();
        if (err_before != cudaSuccess) {
            cudaGetLastError();
        }

        cudaError_t rc = cudaMalloc(&gpu_ptr, Real_size_per_gpu);
        if (rc != cudaSuccess) {
            return {};
        } else {
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_cuda_malloc_time += elapsed_time;

        MultiGPUPointers().push_back(gpu_ptr);

        Vector vector{Pointer{gpu_ptr, &detail::do_not_free}, num_qubits};
        state_parts.emplace_back(std::move(vector));
    }

    GlobalStateParts() = state_parts;
    MPI_Barrier(MPI_COMM_WORLD);
    return state_parts;
}


  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &detail::free}, 0};
  }

  static bool IsNull(const Vector& vector) {
    return vector.get() == nullptr;
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    cudaMemcpy(dest.get(), src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToDevice);

    return true;
  }

  bool Copy(const Vector& src, fp_type* dest) const {
    cudaMemcpy(dest, src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToHost);
    return true;
  }

  bool Copy(const fp_type* src, Vector& dest) const {
    cudaMemcpy(dest.get(), src,
               sizeof(fp_type) * Impl::MinSize(dest.num_qubits()),
               cudaMemcpyHostToDevice);
    return true;
  }

  bool Copy(const fp_type* src, uint64_t size, Vector& dest) const {
    size = std::min(size, Impl::MinSize(dest.num_qubits()));
    cudaMemcpy(dest.get(), src,
               sizeof(fp_type) * size,
               cudaMemcpyHostToDevice);
    return true;
  }


/*
 * InitializeP2P:
 * Enables CUDA Peer-to-Peer (P2P) access between all GPU pairs within a node.
 * Checks P2P capability and activates bidirectional access if supported.
 * This allows direct memory reads across GPUs to reduce intra-node communication overhead.
*/

   static void InitializeP2P(int num_gpus) {
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                        if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        } else if (err != cudaSuccess) {
                                   i, j, cudaGetErrorString(err);
                        } else {
                        }
                    } else {
                    }
                }
            }
        }
    }




static uint64_t MinSize(unsigned num_qubits) {
    num_qubits = 40;  
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    return result;
}



  void DeviceSync() {
    //cudaDeviceSynchronize();
  }

 protected:
};

template <typename Impl, typename FP>
std::vector<typename VectorSpaceCUDA<Impl, FP>::fp_type*> 
VectorSpaceCUDA<Impl, FP>::multi_gpu_ptrs_;

template <typename Impl, typename FP>
double VectorSpaceCUDA<Impl, FP>::total_cuda_malloc_time = 0.0;
}  // namespace qsim

#endif  // VECTORSPACE_CUDA_H_
