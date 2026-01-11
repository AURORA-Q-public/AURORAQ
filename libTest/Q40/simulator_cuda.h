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

#ifndef SIMULATOR_CUDA_H_
#define SIMULATOR_CUDA_H_
#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif



#include "bits.h"
#include "statespace_cuda.h"
#include "simulator_cuda_kernels.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <stdexcept>
#include <bitset>
#include <random>
#include </opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include/mpi.h>
#include <chrono>
#include <unordered_map>
#include <cfloat> 
#include <zstd.h>
#include <chrono>
#include <fcntl.h>    // open, O_RDONLY, O_WRONLY, O_CREAT 등
#include <unistd.h>   // pread, pwrite, close



#define NUM_SUBSETS 8192
#define MAX_GPUS_PER_NODE 4



struct TimerStats {
    double total_time = 0.0;
    size_t call_count = 0;
};

static std::unordered_map<std::string, TimerStats> _Timers;

static std::unordered_set<size_t> _AccessedSubsets;

static std::unordered_set<size_t> _MaterializedSubsets;

static std::mutex _LogMutex;



#define START_CPU_TIMER(name) \
    auto start_##name = std::chrono::high_resolution_clock::now();

#define END_CPU_TIMER(name) \
    { auto end_##name = std::chrono::high_resolution_clock::now(); \
      double elapsed_##name = std::chrono::duration<double>(end_##name - start_##name).count(); \
      _Timers[#name].total_time += elapsed_##name; \
      _Timers[#name].call_count += 1; }

#define START_SUB_TIMER(name) \
    auto start_##name = std::chrono::high_resolution_clock::now();

#define END_SUB_TIMER(name) \
    { auto end_##name = std::chrono::high_resolution_clock::now(); \
      double elapsed_##name = std::chrono::duration<double>(end_##name - start_##name).count(); \
      _Timers[#name].total_time += elapsed_##name; \
      _Timers[#name].call_count += 1; }




namespace qsim {

/**
 * Quantum circuit simulator with GPU vectorization.
 */

    
template <typename FP = float>
class SimulatorCUDA final {
 public:
  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using idx_type = uint64_t;
  using Complex = qsim::Complex<double>;
  std::vector<fp_type*> state_parts_;
  std::vector<std::vector<fp_type>> host_buffers_;
  mutable  std::vector<std::vector<cudaStream_t>> subset_streams;

  std::mutex cpu_table_mtx;
  static std::unordered_map<size_t, std::atomic<bool>> g_flush_in_progress;





  std::vector<StateSpace> state_spaces_;    
  std::vector<ncclComm_t> nccl_comms_;     
  std::vector<char*> d_ws_list_;        
  std::vector<void*> multi_gpu_stream_buffers[128];

  //char* d_ws_list;
  void* scratch_;
  uint64_t scratch_size_;
  mutable State state_; 

  static double total_index_calc_time;
  static double total_apply_gate_time;
  static double total_cpu_time_H;
  static float total_gpu_time_H;
  static double total_cpu_time;
  static float total_gpu_time;
    static double total_cpu_time_copy_h;   
    static float  total_gpu_time_copy_h; 
    static double total_cpu_time_copy_l;   
    static float total_gpu_time_copy_l;

static double total_mpi_bcast_time;

  static size_t MAX_STREAMS_PER_GPU;
  static size_t MAX_CACHE_SIZE;
  static size_t MAX_CPU_CACHE_SIZE;
  
SimulatorCUDA(const std::vector<StateSpace>& state_spaces,
              const std::vector<ncclComm_t>& nccl_comms)
    : state_spaces_(state_spaces),
      nccl_comms_(nccl_comms),
      scratch_size_(0) {

    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Barrier(MPI_COMM_WORLD);
        
    size_t num_gpus = state_spaces_.size();


    unsigned num_qubits = 40;

    size_t total_size = MinSize(num_qubits);

    size_t size_per_node = 0;
    size_t size_per_gpu = 0;
    size_t Real_size_per_gpu = 0;
    
    if (total_nodes > 1) {
        size_per_node = total_size / total_nodes;  
        size_per_gpu = size_per_node / num_gpus;   
        Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

    } else {
        size_per_node = total_size;                
        size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;
    }

        
    d_ws_list_.resize(num_gpus);

    cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
    }

     
    AllocateWorkspace(num_gpus);
    InitSubsetZero(num_qubits);
    InitHostBuffersStructureOnly(num_qubits);
    InitHostCacheConfig(num_qubits);
    subset_streams.resize(num_gpus);
    for (size_t i = 0; i < num_gpus; ++i) {
        subset_streams[i].resize(MAX_STREAMS_PER_GPU, nullptr);
    }

    InitGPUCacheAndStreamConfig(num_qubits);
    InitStreamsForAllGPUs(num_gpus);
    InitMultiGPUBuffers(num_qubits, num_gpus);
        


    }


    
    ~SimulatorCUDA() {
        int world_rank, num_nodes;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  
        MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
        unsigned num_qubits = 40;
        size_t gpus_per_node = d_ws_list_.size();    


    
        printf("DEBUG: Node %d - Starting SimulatorCUDA destructor.\n", world_rank);
    
        // Free workspace memory for each GPU on the current node
        for (size_t i = 0; i < d_ws_list_.size(); ++i) {
            cudaSetDevice(i); // Set the appropriate GPU for the current node only
            if (d_ws_list_[i] != nullptr) {
                cudaError_t err = cudaFree(d_ws_list_[i]);
                if (err != cudaSuccess) {
     
                } else {

                }
            }
        }
    
        // Free scratch memory if allocated (only for the current node)
        if (scratch_ != nullptr) {
            cudaError_t err = cudaFree(scratch_);
            if (err != cudaSuccess) {
                printf("ERROR: Node %d - cudaFree failed for scratch_: %s\n",
                       world_rank, cudaGetErrorString(err));
            } else {
                printf("DEBUG: Node %d - Successfully freed scratch_ memory.\n", world_rank);
            }
        }
    
        MPI_Barrier(MPI_COMM_WORLD);
    }

  

  static constexpr unsigned max_buf_size = 8192 * sizeof(FP)
      + 128 * sizeof(idx_type) + 96 * sizeof(unsigned);
  static constexpr const char* kStoragePath = "./offload_data/";


  char* d_ws;
  char h_ws0[max_buf_size];
  char* h_ws = (char*) h_ws0;



//test Compress / Decompress


std::vector<uint8_t> CompressZstd(const std::vector<fp_type>& input) {
    const size_t input_size = input.size() * sizeof(fp_type);
    const size_t CHUNK_SIZE = 1ULL << 30;  
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(input.data());

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
        printf("ERROR: Failed to create ZSTD_CCtx\n");
        return {};
    }

    std::vector<uint8_t> compressed;
    compressed.reserve(input_size / 2);

    size_t offset = 0;
    while (offset < input_size) {
        size_t this_chunk = std::min(CHUNK_SIZE, input_size - offset);
        size_t bound = ZSTD_compressBound(this_chunk);

        size_t prev_size = compressed.size();
        compressed.resize(prev_size + bound);

        ZSTD_inBuffer inBuf = { src_ptr + offset, this_chunk, 0 };
        ZSTD_outBuffer outBuf = { compressed.data() + prev_size, bound, 0 };

        size_t ret = ZSTD_compressStream2(cctx, &outBuf, &inBuf, ZSTD_e_continue);
        if (ZSTD_isError(ret)) {
            printf("ERROR: Zstd compression failed: %s\n", ZSTD_getErrorName(ret));
            ZSTD_freeCCtx(cctx);
            return {};
        }

        compressed.resize(prev_size + outBuf.pos);
        offset += this_chunk;
    }

    // flush
    size_t flush_bound = ZSTD_CStreamOutSize();
    size_t prev_size = compressed.size();
    compressed.resize(prev_size + flush_bound);
    ZSTD_inBuffer inBuf = { nullptr, 0, 0 };
    ZSTD_outBuffer outBuf = { compressed.data() + prev_size, flush_bound, 0 };
    size_t ret = ZSTD_compressStream2(cctx, &outBuf, &inBuf, ZSTD_e_end);
    if (ZSTD_isError(ret)) {
        printf("ERROR: Zstd flush failed: %s\n", ZSTD_getErrorName(ret));
        ZSTD_freeCCtx(cctx);
        return {};
    }
    compressed.resize(prev_size + outBuf.pos);

    ZSTD_freeCCtx(cctx);
    return compressed;
}

std::vector<fp_type> DecompressZstd(const std::vector<uint8_t>& compressed, size_t original_count) {
    const size_t output_size = original_count * sizeof(fp_type);
    const size_t CHUNK_SIZE = 1ULL << 30; 
    std::vector<fp_type> output(original_count);

    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) {
        printf("ERROR: Failed to create ZSTD_DCtx\n");
        return {};
    }

    size_t in_offset = 0;
    size_t out_offset = 0;
    while (in_offset < compressed.size()) {
        ZSTD_inBuffer inBuf = { compressed.data() + in_offset, compressed.size() - in_offset, 0 };

        while (inBuf.pos < inBuf.size) {
            size_t chunk = std::min(CHUNK_SIZE, output_size - out_offset);
            ZSTD_outBuffer outBuf = { reinterpret_cast<char*>(output.data()) + out_offset, chunk, 0 };

            size_t ret = ZSTD_decompressStream(dctx, &outBuf, &inBuf);
            if (ZSTD_isError(ret)) {
                printf("ERROR: Zstd decompression failed: %s\n", ZSTD_getErrorName(ret));
                ZSTD_freeDCtx(dctx);
                return {};
            }
            out_offset += outBuf.pos;
        }

        in_offset += inBuf.pos;
    }

    ZSTD_freeDCtx(dctx);
    return output;
}


void InitMultiGPUBuffers(unsigned num_qubits, size_t num_gpus) {
    size_t real_size = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    size_t alloc_bytes = real_size * sizeof(fp_type);

    for (size_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        multi_gpu_stream_buffers[gpu].resize(MAX_STREAMS_PER_GPU);

        for (size_t s = 0; s < MAX_STREAMS_PER_GPU; ++s) {
            cudaMalloc(&multi_gpu_stream_buffers[gpu][s], alloc_bytes);
        }
    }

}


size_t QueryGPUMemoryBytes(size_t gpu_id) {
    cudaSetDevice(gpu_id);
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    return total_bytes;
}

size_t CalculateMaxCacheSizePerGPU(size_t gpu_mem_bytes, size_t subset_bytes) {
    const double SAFETY_FACTOR = 0.80;
    const size_t max_cap = 24;
    size_t est = static_cast<size_t>((gpu_mem_bytes * SAFETY_FACTOR) / subset_bytes);

    return std::min(est, max_cap);
}

size_t EstimateStreamsPerGPU(size_t gpu_mem_bytes, size_t subset_bytes) {
    const size_t practical_limit = 8;
    size_t result = std::min(gpu_mem_bytes / subset_bytes, practical_limit);

    return result;
}

void InitGPUCacheAndStreamConfig(unsigned num_qubits, size_t gpu_id_sample = 0) {
    size_t subset_elements = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    size_t subset_bytes = subset_elements * sizeof(fp_type);
    size_t gpu_mem_bytes = QueryGPUMemoryBytes(gpu_id_sample);

    MAX_CACHE_SIZE = CalculateMaxCacheSizePerGPU(gpu_mem_bytes, subset_bytes);
    MAX_STREAMS_PER_GPU = EstimateStreamsPerGPU(gpu_mem_bytes, subset_bytes);

}

size_t CalculateMaxCPUCacheSize(size_t subset_bytes) {
    const size_t SYSTEM_RAM_BYTES = size_t(256UL) * 1024 * 1024 * 1024;
    const double SAFETY_FACTOR = 0.80;

    size_t max_cacheable_bytes = static_cast<size_t>(SYSTEM_RAM_BYTES * SAFETY_FACTOR);
    size_t est = max_cacheable_bytes / subset_bytes;

    const size_t HARD_CAP = 64;

    return std::min(est, HARD_CAP);
}

void InitHostCacheConfig(unsigned num_qubits) {
    size_t subset_elements = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    size_t subset_bytes = subset_elements * sizeof(fp_type);
    MAX_CPU_CACHE_SIZE = CalculateMaxCPUCacheSize(subset_bytes);


}

void InitStreamsForAllGPUs(size_t num_gpus) {
    if (subset_streams.size() != num_gpus) {
        subset_streams.resize(num_gpus);
        for (auto& vec : subset_streams) {
            vec.resize(MAX_STREAMS_PER_GPU, nullptr);
        }
    }

    for (size_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);
        MAX_STREAMS_PER_GPU = 2;
        for (size_t s = 0; s < MAX_STREAMS_PER_GPU; ++s) {
            cudaStreamCreate(&subset_streams[gpu][s]);
        }
    }

}

void AllocateWorkspace(size_t num_gpus) {
    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    for (unsigned i = 0; i < num_gpus; ++i) {
        cudaError_t status = cudaSetDevice(i);
        if (status != cudaSuccess) {
            printf("ERROR: Failed to set device %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }

        size_t free_mem = 0, total_mem = 0;
        status = cudaMemGetInfo(&free_mem, &total_mem);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMemGetInfo failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }

        status = cudaMalloc(&d_ws_list_[i], max_buf_size);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMalloc failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            d_ws_list_[i] = nullptr;
            continue;
        }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
           printf("ERROR: cudaDeviceSynchronize failed on GPU %u: %s\n", i, cudaGetErrorString(status));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

}



struct SubsetCacheEntry {
    size_t subset_id;
    double last_used_time;  
    bool is_dirty;
    size_t access_count;    
};

struct CPUCacheEntry {
    std::vector<fp_type> data;   
    double last_used_time;     
    bool is_dirty;            
    size_t access_count;        
    double reuse_score;         
};

struct CPUBufferEntry {
    std::vector<fp_type> data;
    bool is_dirty = false;
    bool flush_in_progress = false; 
};



struct GpuEntrySnapshot {
    size_t subset_id;
    bool   is_dirty;
    double last_used_time;
    size_t access_count;
};



mutable std::unordered_map<size_t, SubsetCacheEntry> gpu_subset_cache_table[128];
mutable std::unordered_map<size_t, CPUCacheEntry> cpu_subset_buffer_table;
mutable  std::unordered_map<size_t, size_t> subset_to_gpu_table;




std::unordered_map<size_t, std::vector<fp_type>> cpu_double_buffers[2];
std::atomic<size_t> active_buffer_index = 0;


double GetCurrentTimeMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}
void UpdateCacheEntry(size_t gpu_id, size_t subset_id) {
    auto& entry = gpu_subset_cache_table[gpu_id][subset_id];
    entry.last_used_time = GetCurrentTimeMs();
    entry.access_count += 1;

}

size_t SelectLRUVictim(size_t gpu_id) {
    double oldest = DBL_MAX;
    size_t victim = static_cast<size_t>(-1);

    for (const auto& [sid, entry] : gpu_subset_cache_table[gpu_id]) {
        if (entry.last_used_time < oldest) {
            oldest = entry.last_used_time;
            victim = sid;
        }
    }

    return victim;
}

std::optional<size_t> SelectLFUVictim(size_t gpu_id) {
    double min_score = DBL_MAX;
    std::optional<size_t> victim;

    double now = GetCurrentTimeMs();

    for (const auto& [sid, entry] : gpu_subset_cache_table[gpu_id]) {
        if (sid == 0) continue;  

        double reuse_score = static_cast<double>(entry.access_count) /
                             (now - entry.last_used_time + 1e-6);

        if (reuse_score < min_score) {
            min_score = reuse_score;
            victim = sid;
        }
    }

    return victim;
}


void EvictLFUVictim(size_t gpu_id, unsigned num_qubits,
                    cudaStream_t stream, size_t stream_id) {
    auto victim_opt = SelectLFUVictim(gpu_id);
    if (!victim_opt) {
        return;
    }

    size_t victim_sid = *victim_opt;
    auto it = gpu_subset_cache_table[gpu_id].find(victim_sid);
    if (it == gpu_subset_cache_table[gpu_id].end()) {
        return;
    }

    GpuEntrySnapshot snap {
        .subset_id      = victim_sid,
        .is_dirty       = it->second.is_dirty,
        .last_used_time = it->second.last_used_time,
        .access_count   = it->second.access_count
    };

    if (snap.is_dirty) {

        EvictToCPU(num_qubits, snap, gpu_id, stream, stream_id);

    } else {
        gpu_subset_cache_table[gpu_id].erase(victim_sid);
        subset_to_gpu_table.erase(victim_sid);
    }

}

void EvictToCPU(unsigned num_qubits, const GpuEntrySnapshot snap, size_t gpu_id,
                cudaStream_t stream, size_t stream_id) {
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    void* device_ptr = multi_gpu_stream_buffers[gpu_id][stream_id];

    size_t buf_idx = active_buffer_index.load();
    std::vector<fp_type>& temp = cpu_double_buffers[buf_idx][snap.subset_id];
    temp.resize(Real_size_per_subset);

    cudaMemcpyAsync(temp.data(), device_ptr,
                    Real_size_per_subset * sizeof(fp_type),
                    cudaMemcpyDeviceToHost, stream);

    cudaEvent_t done;
    cudaEventCreate(&done);
    cudaEventRecord(done, stream);

    auto callback = [=]() {
        double now = GetCurrentTimeMs();
        double reuse_score = static_cast<double>(snap.access_count) /
                             (now - snap.last_used_time + 1e-6);

        
        cpu_subset_buffer_table[snap.subset_id] = {
            .data          = std::move(cpu_double_buffers[buf_idx][snap.subset_id]),
            .last_used_time= now,
            .is_dirty      = snap.is_dirty,      
            .access_count  = snap.access_count,
            .reuse_score   = reuse_score
        };


        gpu_subset_cache_table[gpu_id].erase(snap.subset_id);
        subset_to_gpu_table.erase(snap.subset_id);

        if (cpu_subset_buffer_table.size() > MAX_CPU_CACHE_SIZE) {
            size_t min_sid = size_t(-1);
            double min_score = DBL_MAX;

            for (const auto& [sid, entry] : cpu_subset_buffer_table) {
                if (sid == 0) continue;
                if (entry.reuse_score < min_score) {
                    min_score = entry.reuse_score;
                    min_sid = sid;
                }
            }

            if (min_sid != size_t(-1)) {
                if (cpu_subset_buffer_table[min_sid].is_dirty) {
                    FlushCPUBufferToDisk(min_sid, num_qubits);
                }
                cpu_subset_buffer_table.erase(min_sid);
            }
        }
    };

    std::thread([done, callback]() {
        cudaEventSynchronize(done);
        callback();
        cudaEventDestroy(done);
    }).detach();

}


inline size_t GetRealSizePerGPU(unsigned num_qubits, int total_nodes, size_t num_gpus) {
    size_t total_size = MinSize(num_qubits);
    size_t size_per_node = total_size / total_nodes;
    size_t size_per_gpu = size_per_node / num_gpus;
    size_t real_size = ((size_per_gpu + 128) / 128) * 128;

    return real_size;
}

inline size_t GetRealSizePerSubset(unsigned num_qubits, size_t num_subsets) {
    size_t total_size = MinSize(num_qubits);
    size_t subset_size = total_size / num_subsets;
    size_t real_size = ((subset_size + 128) / 128) * 128;

    return real_size;
}

void InitAndFlushStorageSubsets_Async(unsigned num_qubits) {
    using namespace std::chrono;
    const size_t MAX_INFLIGHT = 32;
    int node_id, total_nodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);

    size_t num_subsets = NUM_SUBSETS;
    size_t subsets_per_node = num_subsets / total_nodes;
    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, num_subsets);
;

    auto start_total = high_resolution_clock::now();

    for (size_t offset = 0; offset < subsets_per_node; offset += MAX_INFLIGHT) {
        size_t current_batch_size = std::min(MAX_INFLIGHT, subsets_per_node - offset);

        std::vector<std::vector<fp_type>> buffers(current_batch_size);
        std::vector<MPI_Request> requests(current_batch_size);
        std::vector<MPI_File> files(current_batch_size);
        std::vector<size_t> subset_ids(current_batch_size);

        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t subset_id = node_id * subsets_per_node + offset + i;
            subset_ids[i] = subset_id;
            buffers[i].resize(Real_size_per_subset, 0.0f);

            if (subset_id == 0) {
                buffers[i][0] = 1.0f;
            }

            std::string filename = std::string(kStoragePath) + "node" +
                                   std::to_string(node_id) + "_subset" +
                                   std::to_string(subset_id) + ".bin";

            MPI_File_open(MPI_COMM_SELF, filename.c_str(),
                          MPI_MODE_CREATE | MPI_MODE_WRONLY,
                          MPI_INFO_NULL, &files[i]);

            MPI_File_iwrite_at(files[i], 0,
                               buffers[i].data(),
                               Real_size_per_subset,
                               MPI_FLOAT, &requests[i]);
        }

        for (size_t i = 0; i < current_batch_size; ++i) {
            MPI_Status status;
            MPI_Wait(&requests[i], &status);
            MPI_File_close(&files[i]);

        }

        for (auto& buf : buffers) {
            buf.clear();
            buf.shrink_to_fit();
        }
    }

    auto end_total = high_resolution_clock::now();
    double elapsed_ms = duration<double, std::milli>(end_total - start_total).count();

    MPI_Barrier(MPI_COMM_WORLD);

}



void InitSubsetZero(unsigned num_qubits) {
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    std::vector<fp_type> buffer(Real_size_per_subset, 0.0f);

    buffer[0] = 1.0f; 

    std::string filename = std::string(kStoragePath) + "node" +
                           std::to_string(node_id) + "_subset0.bin";

    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        return;
    }
    fwrite(buffer.data(), sizeof(fp_type), Real_size_per_subset, fp);
    fclose(fp);

}



void InitHostBuffersStructureOnly(unsigned num_qubits) {
    int node_id, total_nodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);

    size_t num_subsets = NUM_SUBSETS;
    size_t subsets_per_node = num_subsets / total_nodes;
    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, num_subsets);

    if (host_buffers_.size() < num_subsets)
        host_buffers_.resize(num_subsets);  

}


void FetchSubsetToGPU(unsigned num_qubits, size_t subset_idx, size_t gpu_id,
                      cudaStream_t stream, size_t stream_id) {
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    void* device_ptr = multi_gpu_stream_buffers[gpu_id][stream_id];

    if (gpu_subset_cache_table[gpu_id].count(subset_idx)) {
        START_SUB_TIMER(Fetch_CacheHit);
        UpdateCacheEntry(gpu_id, subset_idx);
        END_SUB_TIMER(Fetch_CacheHit);
        return;
    }

    if (gpu_subset_cache_table[gpu_id].size() >= MAX_CACHE_SIZE) {
        START_SUB_TIMER(Fetch_Eviction);
        EvictLFUVictim(gpu_id, num_qubits, stream, stream_id);
        END_SUB_TIMER(Fetch_Eviction);
    }

    std::vector<fp_type> temp_buffer(Real_size_per_subset, 0.0f);
    std::string filename = std::string(kStoragePath) + "node" +
                           std::to_string(node_id) + "_subset" +
                           std::to_string(subset_idx) + ".bin";

    bool loaded_from_disk = false;
    if (FILE* fp = fopen(filename.c_str(), "rb")) {
        START_SUB_TIMER(Fetch_IORead);
        size_t n = fread(temp_buffer.data(), sizeof(fp_type), Real_size_per_subset, fp);
        fclose(fp);
        loaded_from_disk = (n == Real_size_per_subset);
        END_SUB_TIMER(Fetch_IORead);
    }

    if (!loaded_from_disk) {
        START_SUB_TIMER(Fetch_LazyInit);
        if (subset_idx == 0) temp_buffer[0] = 1.0f;  
        END_SUB_TIMER(Fetch_LazyInit);
    }

    START_SUB_TIMER(Fetch_GPUCopy);
    cudaSetDevice(gpu_id);
    cudaMemcpyAsync(device_ptr,
                    temp_buffer.data(),
                    Real_size_per_subset * sizeof(fp_type),
                    cudaMemcpyHostToDevice, stream);
    END_SUB_TIMER(Fetch_GPUCopy);

    gpu_subset_cache_table[gpu_id][subset_idx] = {
        .subset_id      = subset_idx,
        .last_used_time = GetCurrentTimeMs(),
        .is_dirty       = !loaded_from_disk,   
        .access_count   = 1
    };

}


void FlushCPUBufferToDisk(size_t subset_idx, unsigned num_qubits) {
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    if (cpu_subset_buffer_table.count(subset_idx) == 0) {
        return;
    }
    const auto& entry = cpu_subset_buffer_table[subset_idx];
    if (!entry.is_dirty) {
        return;
    }

    std::string filename = std::string(kStoragePath) + "node" +
                           std::to_string(node_id) + "_subset" +
                           std::to_string(subset_idx) + ".bin";

    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        printf(" ERROR: Cannot open %s for writing\n", filename.c_str());
        return;
    }

    fwrite(entry.data.data(), sizeof(fp_type), entry.data.size(), fp);
    fclose(fp);


    cpu_subset_buffer_table[subset_idx].is_dirty = false;

}


size_t SelectGPUForSubset(size_t subset_id, size_t num_gpus_per_node) {
    if (subset_id == 0) {
        return 0;
    }

    if (subset_to_gpu_table.count(subset_id)) {
        return subset_to_gpu_table[subset_id];
    }

    size_t min_total = SIZE_MAX;
    size_t best_gpu = 1;

    for (size_t gpu_id = 1; gpu_id < num_gpus_per_node; ++gpu_id) {
        size_t cache_size = gpu_subset_cache_table[gpu_id].size();
        size_t sum_access = 0;

        for (auto& [sid, entry] : gpu_subset_cache_table[gpu_id]) {
            sum_access += entry.access_count;
        }


        if (sum_access < min_total) {
            min_total = sum_access;
            best_gpu = gpu_id;
        }
    }


    subset_to_gpu_table[subset_id] = best_gpu;

    return best_gpu;
}


static uint64_t MinSize(unsigned num_qubits) {
    num_qubits = 40;
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    return result;
      
};




void SetAllZeros() {
    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus = multi_gpu_pointers.size();
    unsigned num_qubits = 40;
    size_t total_size = MinSize(num_qubits);

    size_t size_per_node = 0;
    size_t size_per_gpu = 0;
    size_t Real_size_per_gpu = 0;
    
    if (total_nodes > 1) {
        size_per_node = total_size / total_nodes;  
        size_per_gpu = size_per_node / num_gpus;   
        Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

    } else {
        size_per_node = total_size;              
        size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;
    }


    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMemset(multi_gpu_pointers[i], 0, Real_size_per_gpu);
    }

    if(node_id == 0){
    fp_type one[1] = {1};
    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(0);
        cudaMemcpy(multi_gpu_pointers[0], one, sizeof(fp_type), cudaMemcpyHostToDevice);
 
       }
    }
}
 

std::vector<size_t> CalculateAffectedIndices(
    size_t num_qubits, const std::vector<unsigned>& qs,
    size_t num_nodes, size_t num_gpus_per_node, size_t size_per_gpu) const {

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    size_t total_gpus = num_nodes * num_gpus_per_node;
    size_t size_per_node = size_per_gpu * num_gpus_per_node;
    size_t num_indices = 1ULL << qs.size();

    std::vector<size_t> indices_global;
    indices_global.reserve(num_indices);

    std::vector<size_t> xs(qs.size());
    for (size_t i = 0; i < qs.size(); ++i) {
        xs[i] = size_t{1} << qs[i];
    }

    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        size_t node_offset = node_id * size_per_node;
        for (size_t i = 0; i < num_indices; ++i) {
            size_t index = 0;
            for (size_t j = 0; j < qs.size(); ++j) {
                index += xs[j] * ((i >> j) & 1);
            }
            index += node_offset;

            if (index < node_offset || index >= node_offset + size_per_node) {
                continue;
            }

            size_t index_in_node = index % size_per_node;
            size_t gpu_id_local = index_in_node / size_per_gpu;
            size_t local_index = index_in_node % size_per_gpu;
            size_t global_gpu_id = node_id * num_gpus_per_node + gpu_id_local;

            if (global_gpu_id >= total_gpus) {
                continue;
            }


            indices_global.push_back(index);
        }
    }

    auto end_time = high_resolution_clock::now();
    double elapsed_time = duration<double, std::milli>(end_time - start_time).count();
    total_index_calc_time += elapsed_time;


    return indices_global;
}


std::unordered_set<size_t> CalculateAffectedSubsetsSimple(
    unsigned num_qubits, const std::vector<unsigned>& qs,
    size_t num_subsets) const {

    size_t total_size = MinSize(num_qubits);
    size_t logical_subset_size = total_size / NUM_SUBSETS;
    size_t padded_subset_size = ((logical_subset_size + 128) / 128) * 128;

    std::unordered_set<size_t> subset_ids;
    size_t num_affected = 1ULL << qs.size();

    for (size_t i = 0; i < num_affected; ++i) {
        size_t index = 0;
        for (size_t j = 0; j < qs.size(); ++j) {
            index |= ((i >> j) & 1ULL) << qs[j];
        }
        size_t subset_id = index / padded_subset_size;
        if (subset_id < NUM_SUBSETS) subset_ids.insert(subset_id);
    }

    static std::unordered_set<size_t> previous_subset_ids;
    std::vector<size_t> reused_ids;
    size_t reuse_count = 0;

    for (size_t sid : subset_ids) {
        if (previous_subset_ids.count(sid)) {
            reused_ids.push_back(sid);
            ++reuse_count;
        }
    }


    previous_subset_ids = subset_ids;

    return subset_ids;
}


inline size_t GetOwnerNode(size_t subset_id, size_t total_nodes) {
    size_t per_node = (NUM_SUBSETS + total_nodes - 1) / total_nodes;  
    return subset_id / per_node;
}


void ApplyGate_(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t num_nodes = world_size;
    size_t num_gpus_per_node = MAX_GPUS_PER_NODE;
    size_t total_gpus = num_nodes * num_gpus_per_node;

    unsigned num_qubits = 40;
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = total_size / total_gpus;
    size_t Real_size_per_subset = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);

    std::vector<size_t> affected_indices;
    if (world_rank == 0) {
        affected_indices = CalculateAffectedIndices(num_qubits, qs, num_nodes, num_gpus_per_node, size_per_gpu);
    }


    
    size_t indices_size = affected_indices.size();
    size_t total_bytes = indices_size * sizeof(uint64_t);
    static size_t total_transmitted_bytes = 0;
    total_transmitted_bytes += total_bytes;
    
    MPI_Bcast(&indices_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    auto t_start_mpi = std::chrono::high_resolution_clock::now();

    
    if (world_rank != 0) affected_indices.resize(indices_size);
    MPI_Bcast(affected_indices.data(), indices_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    auto t_end_mpi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mpi_diff = t_end_mpi - t_start_mpi;
    
    total_mpi_bcast_time += mpi_diff.count(); 

    
    std::unordered_set<size_t> affected_subsets = CalculateAffectedSubsetsSimple(num_qubits, qs, NUM_SUBSETS);

    std::vector<size_t> local_subsets;
    for (auto sid : affected_subsets) {
        if (GetOwnerNode(sid, world_size) == static_cast<size_t>(world_rank)) {
            local_subsets.push_back(sid);
        }
    }


    for (size_t subset_id : local_subsets) {
        size_t gpu_id = (subset_id == 0) ? 0 : SelectGPUForSubset(subset_id, num_gpus_per_node);
        size_t stream_id = subset_id % MAX_STREAMS_PER_GPU;
        cudaStream_t stream = subset_streams[gpu_id][stream_id];
        void* device_ptr = multi_gpu_stream_buffers[gpu_id][stream_id];  

        if (!gpu_subset_cache_table[gpu_id].count(subset_id)) {
            FetchSubsetToGPU(num_qubits, subset_id, gpu_id, stream, stream_id);
        } else {
            UpdateCacheEntry(gpu_id, subset_id);
        }

        if (qs.size() == 0) {
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream);
        } else if (qs[0] > 4) {
            switch (qs.size()) {
                case 1: const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 2: const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 3: const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 4: const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 5: const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 6: const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
            }
        } else {
            switch (qs.size()) {
                case 1: const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 2: const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 3: const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 4: const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 5: const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
                case 6: const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state, gpu_id, subset_id, device_ptr, stream); break;
            }
        }
    }

    static size_t gate_call_counter = 0;
    if (++gate_call_counter % 100 == 0) {
        for (size_t gpu_id = 0; gpu_id < 128; ++gpu_id) {
            for (auto& it : gpu_subset_cache_table[gpu_id]) {
                it.second.access_count = static_cast<uint64_t>(it.second.access_count * 0.5);
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    double elapsed_time = duration<double, std::milli>(end_time - start_time).count();
    total_apply_gate_time += elapsed_time;
}


void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
  const_cast<SimulatorCUDA*>(this)->ApplyGate_(qs, matrix, state);
}


  /**
   * Applies a controlled gate using CUDA instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
   
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    printf("ApplyControlledGate\n");
    if (cqs.size() == 0) {
      //ApplyGate(qs, matrix, state);
      return;
    }

    if (cqs[0] < 5) {
      switch (qs.size()) {
      case 0:
        ApplyControlledGateL<0>(qs, cqs, cvals, matrix, state);
        break;
      case 1:
        ApplyControlledGateL<1>(qs, cqs, cvals, matrix, state);
        break;
      case 2:
        ApplyControlledGateL<2>(qs, cqs, cvals, matrix, state);
        break;
      case 3:
        ApplyControlledGateL<3>(qs, cqs, cvals, matrix, state);
        break;
      case 4:
        ApplyControlledGateL<4>(qs, cqs, cvals, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    } else {
      if (qs.size() == 0) {
        ApplyControlledGateHH<0>(qs, cqs, cvals, matrix, state);
      } else if (qs[0] > 4) {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      } else {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateLH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateLH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateLH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateLH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      }
    }
  }
  

  /**
   * Computes the expectation value of an operator using CUDA instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    printf("ExpectationValue\n");

    if (qs[0] > 4) {
      switch (qs.size()) {
      case 1:
        return ExpectationValueH<1>(qs, matrix, state);
      case 2:
        return ExpectationValueH<2>(qs, matrix, state);
      case 3:
        return ExpectationValueH<3>(qs, matrix, state);
      case 4:
        return ExpectationValueH<4>(qs, matrix, state);
      case 5:
        return ExpectationValueH<5>(qs, matrix, state);
      case 6:
        return ExpectationValueH<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
    } else {
      switch (qs.size()) {
      case 1:
        return ExpectationValueL<1>(qs, matrix, state);
      case 2:
        return ExpectationValueL<2>(qs, matrix, state);
      case 3:
        return ExpectationValueL<3>(qs, matrix, state);
      case 4:
        return ExpectationValueL<4>(qs, matrix, state);
      case 5:
        return ExpectationValueL<5>(qs, matrix, state);
      case 6:
        return ExpectationValueL<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 32;
  }

 private:

  template <unsigned G>
  struct IndicesH {
    static constexpr unsigned gsize = 1 << G;
    unsigned int gsize_t;
    static constexpr unsigned matrix_size = 2 * gsize * gsize * sizeof(fp_type);
    static constexpr unsigned xss_size = 32 * sizeof(idx_type) * (1 + (G == 6));
    static constexpr unsigned ms_size = 32 * sizeof(idx_type);
    static constexpr unsigned xss_offs = matrix_size;
    static constexpr unsigned ms_offs = xss_offs + xss_size;
    static constexpr unsigned buf_size = ms_offs + ms_size;


    IndicesH(char* p)
        : xss((idx_type*) (p + xss_offs)), ms((idx_type*) (p + ms_offs)) {}


    idx_type* xss;
    idx_type* ms;
  };

  template <unsigned G>
  struct IndicesL : public IndicesH<G> {
    using Base = IndicesH<G>;
    static constexpr unsigned qis_size = 32 * sizeof(unsigned) * (1 + (G == 6));
    static constexpr unsigned tis_size = 32 * sizeof(unsigned);
    static constexpr unsigned qis_offs = Base::buf_size;
    static constexpr unsigned tis_offs = qis_offs + qis_size;
    static constexpr unsigned buf_size = tis_offs + tis_size;


          
    IndicesL(char* p)
      : Base(p), qis((unsigned*) (p + qis_offs)),
         tis((unsigned*) (p + tis_offs)) {}


    unsigned* qis;
    unsigned* tis;
  };


  template <unsigned G>
  struct IndicesLC : public IndicesL<G> {
    using Base = IndicesL<G>;
    static constexpr unsigned cis_size = 32 * sizeof(idx_type);
    static constexpr unsigned cis_offs = Base::buf_size;
    static constexpr unsigned buf_size = cis_offs + cis_size;

    IndicesLC(char* p) : Base(p), cis((idx_type*) (p + cis_offs)) {}

    idx_type* cis;
  };

  struct DataC {
    idx_type cvalsh;
    unsigned num_aqs;
    unsigned num_effective_qs;
    unsigned remaining_low_cqs;
  };





struct StateVectorLocation {
    size_t start_index;
    size_t end_index;
    int gpu_id;
    int node_rank;
};

std::unordered_map<size_t, StateVectorLocation> state_location_table;
std::vector<StateVectorLocation> state_location_vector;
int num_gpus_per_node;
int num_nodes;

void InitializeStateLocationTable(int num_gpus, int num_nodes, size_t total_size) {
    size_t chunk_size = total_size / (num_gpus * num_nodes);
    for (int node = 0; node < num_nodes; ++node) {
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            int index = node * num_gpus + gpu;
            StateVectorLocation loc = {
                index * chunk_size,
                (index + 1) * chunk_size - 1,
                gpu,
                node
            };
            state_location_table[loc.start_index] = loc;
            state_location_vector.push_back(loc);  
        }
    }
}

/* 
 * Adaptive Kernel Parameter Adjustment:
 * These functions dynamically configure CUDA block and thread parameters 
 * based on the number of qubits and gate width (G). 
 * To improve execution efficiency and memory safety, ScaleQsim enables 
 * safe mode for large-scale gates or high-qubit circuits. 
 * In safe mode, the number of blocks is constrained by real-time 
 * available GPU memory to avoid over-allocation and OOM errors. 
 * This adaptive mechanism ensures performance across varying workload size. 
*/


template <unsigned G>
void parameterConf_H(unsigned num_qubits, int gpuID,
                     unsigned long long& blocks, unsigned& threads) {
    unsigned k = 5 + G;
    unsigned n = (num_qubits > k) ? (num_qubits - k) : 0;
    unsigned long long size = 1ULL << n;
    threads = 64U;
    unsigned long long max_blocks = (1ULL << 30);

    // 기존 early-return 로직 유지
    bool safe_mode = (num_qubits >= 44 || G >= 7);

    if (!safe_mode) {
        blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    } else {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (gpuID >= device_count) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        cudaSetDevice(gpuID);
        cudaFree(0);

        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        size_t mem_per_thread = 2 * sizeof(fp_type);
        size_t mem_per_block = threads * mem_per_thread;

        unsigned long long mem_bound_blocks = free_mem / mem_per_block;
        unsigned long long size_bound_blocks = std::max(1ULL, size / threads);

        blocks = std::min({max_blocks, mem_bound_blocks, size_bound_blocks});
    }


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuID);
    unsigned SM = prop.multiProcessorCount;

    unsigned long long sm_limit_blocks =
        (unsigned long long)(SM / MAX_STREAMS_PER_GPU) * 2ULL; 

    if (sm_limit_blocks == 0) sm_limit_blocks = 1;

    blocks = std::min(blocks, sm_limit_blocks);

    if (blocks < 1ULL) blocks = 1ULL;
}



void parameterConf_L(unsigned num_qubits, unsigned num_effective_qs, int gpuID,
                     unsigned long long& blocks, unsigned& threads) {

    unsigned k = 5 + num_effective_qs;
    unsigned n = (num_qubits > k) ? (num_qubits - k) : 0;
    unsigned long long size = 1ULL << n;
    threads = 32;
    unsigned long long max_blocks = (1ULL << 30);

    bool safe_mode = (num_qubits >= 44 || num_effective_qs >= 7);

    if (!safe_mode) {
        blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    } else {
        cudaSetDevice(gpuID);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        size_t mem_per_thread = 2 * sizeof(fp_type);
        size_t mem_per_block = threads * mem_per_thread;

        unsigned long long mem_bound_blocks = free_mem / mem_per_block;
        unsigned long long size_bound_blocks = std::max(1ULL, size / threads);

        blocks = std::min({max_blocks, mem_bound_blocks, size_bound_blocks});
    }


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuID);
    unsigned SM = prop.multiProcessorCount;

    unsigned long long sm_limit_blocks =
        (unsigned long long)(SM / MAX_STREAMS_PER_GPU) * 2ULL;

    if (sm_limit_blocks == 0) sm_limit_blocks = 1;

    blocks = std::min(blocks, sm_limit_blocks);

    if (blocks < 1ULL) blocks = 1ULL;
}



template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state,
                size_t gpu_id, size_t subset_id, void* device_ptr, cudaStream_t stream) {

    unsigned num_qubits = 40;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t num_nodes = world_size;
    size_t num_gpus_per_node = MAX_GPUS_PER_NODE;
    size_t total_gpus = num_nodes * num_gpus_per_node;

    size_t total_size = MinSize(num_qubits);
    size_t subset_real_size = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    size_t global_gpu_id = world_rank * num_gpus_per_node + gpu_id;
    size_t global_offset = subset_id * subset_real_size;

    InitializeStateLocationTable(num_gpus_per_node, num_nodes, total_size);
    cudaSetDevice(gpu_id);

    IndicesH<G> h_i(h_ws);
    GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);

    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice, stream);

    IndicesH<G> d_i(d_ws_list_[gpu_id]);

    unsigned long long blocks;
    unsigned threads;
    parameterConf_H<G>(num_qubits, gpu_id, blocks, threads);


    ApplyGateH_Kernel<G><<<blocks, threads, 0, stream>>>(
        (fp_type*) d_ws_list_[gpu_id],
        d_i.xss,
        d_i.ms,
        (fp_type*) device_ptr,
        global_offset,
        subset_real_size,
        gpu_id,
        world_size,
        global_gpu_id
    );


    gpu_subset_cache_table[gpu_id][subset_id].is_dirty = true;

}

template <unsigned G>
void ApplyGateL(const std::vector<unsigned>& qs, const fp_type* matrix, State& state,
                size_t gpu_id, size_t subset_id, void* device_ptr, cudaStream_t stream) {

    unsigned num_qubits = 40;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t num_nodes = world_size;
    size_t num_gpus_per_node = MAX_GPUS_PER_NODE;
    size_t total_gpus = num_nodes * num_gpus_per_node;

    size_t total_size = MinSize(num_qubits);
    size_t subset_real_size = GetRealSizePerSubset(num_qubits, NUM_SUBSETS);
    size_t global_gpu_id = world_rank * num_gpus_per_node + gpu_id;
    size_t global_offset = subset_id * subset_real_size;

    cudaSetDevice(gpu_id);

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL_m(num_qubits, qs, h_i, global_offset);

    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice, stream);

    IndicesL<G> d_i(d_ws_list_[gpu_id]);

    unsigned long long blocks;
    unsigned threads;
    parameterConf_L(num_qubits, num_effective_qs, gpu_id, blocks, threads);


    ApplyGateL_Kernel<G><<<blocks, threads, 0, stream>>>(
        (fp_type*) d_ws_list_[gpu_id],
        d_i.xss,
        d_i.ms,
        d_i.qis,
        d_i.tis,
        1 << num_effective_qs,
        (fp_type*) device_ptr,        
        global_offset,
        subset_real_size,
        gpu_id,
        world_size,
        global_gpu_id
    );


    gpu_subset_cache_table[gpu_id][subset_id].is_dirty = true;

}


    
  template <unsigned G>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, idx_type cvals,
                             const fp_type* matrix, State& state) const {
    unsigned aqs[64];
    idx_type cmaskh = 0;
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);

    unsigned num_aqs = GetHighQubits(qs, 0, cqs, 0, 0, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, h_i.ms);
    GetXss(num_qubits, qs, qs.size(), h_i.xss);

    idx_type cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);
    printf(": ApplyControlledGateHH start\n");
    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);

        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    IndicesH<G> d_i(d_ws);

    ApplyControlledGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_aqs + 1, cvalsh, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateLH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto d = GetIndicesLC(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    printf(": ApplyControlledGateLH start\n");

        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesL<G> d_i(d_ws);

    ApplyControlledGateLH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesLC<G> h_i(h_ws);
    auto d = GetIndicesLCL(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesLC<G> d_i(d_ws);

    ApplyControlledGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis, d_i.cis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs,
        1 << (5 - d.remaining_low_cqs), state.get());
  }

  template <unsigned G>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    //ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
        
    printf(": ExpectationValueH - cudaMemcpyAsync\n");
    fflush(stdout);
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 14 ? n - 14 : 0, 4U);
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, (size / 2) >> s);
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;

    printf(": ExpectationValueH - AllocScratch");
    //fflush(stdout);

    IndicesH<G> d_i(d_ws);

    ExpectationValueH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_iterations_per_block,
        state.get(), Plus<double>(), d_res1);

    printf(": ExpectationValueH - after Kernel Execution");
    
    double mul = size == 1 ? 0.5 : 1.0;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned G>
  std::complex<double> ExpectationValueL(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    //ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    printf(": ExpectationValueL - cudaMemcpyAsync\n");

    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 13 ? n - 13 : 0, 5U);
    unsigned threads = 32;
    unsigned blocks = size >> s;
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;
    
    printf(": ExpectationValueL - AllocScratch");

    IndicesL<G> d_i(d_ws);

    ExpectationValueL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        num_iterations_per_block, state.get(), Plus<double>(), d_res1);
        
    printf(": ExpectationValueL - after Kernel Execution");


    double mul = double(1 << (5 + num_effective_qs - G)) / 32;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned m>
  std::complex<double> ExpectationValueReduceFinal(
      unsigned blocks, double mul,
      const Complex* d_res1, Complex* d_res2) const {
    Complex res2[m];
    
    printf(": ExpectationValueReduceFinal start\n");

    if (blocks <= 16) {
      //ErrorCheck(
      cudaMemcpy(res2, d_res1, blocks * sizeof(Complex), cudaMemcpyDeviceToHost);
      printf(": cudaMemcpyD2H right?\n ");

    } else {
      unsigned threads2 = std::min(1024U, blocks);
      unsigned blocks2 = std::min(m, blocks / threads2);

      unsigned dblocks = std::max(1U, blocks / (blocks2 * threads2));
      unsigned bytes = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<blocks2, threads2, bytes>>>(
          dblocks, blocks, Plus<Complex>(), Plus<double>(), d_res1, d_res2);

      //ErrorCheck(
      cudaMemcpy(res2, d_res2, blocks2 * sizeof(Complex),
                            cudaMemcpyDeviceToHost);
      
      printf(": cudaMemcpyD2H right?\n ");

      blocks = blocks2;
    }

    double re = 0;
    double im = 0;

    for (unsigned i = 0; i < blocks; ++i) {
      re += res2[i].re;
      im += res2[i].im;
    }

    return {mul * re, mul * im};
  }

  template <typename AQ>
  unsigned GetHighQubits(const std::vector<unsigned>& qs, unsigned qi,
                         const std::vector<unsigned>& cqs, unsigned ci,
                         unsigned ai, idx_type& cmaskh, AQ& aqs) const {
    while (1) {
      if (qi < qs.size() && (ci == cqs.size() || qs[qi] < cqs[ci])) {
        aqs[ai++] = qs[qi++];
      } else if (ci < cqs.size()) {
        cmaskh |= idx_type{1} << cqs[ci];
        aqs[ai++] = cqs[ci++];
      } else {
        break;
      }
    }
    
    printf(": GetHighQubits- ai: %u\n", ai);

    return ai;
  }

  template <typename QS>
  void GetMs(unsigned num_qubits, const QS& qs, unsigned qs_size,
             idx_type* ms) const {
    printf(": GetMs start\n");
    if (qs_size == 0) {
      ms[0] = idx_type(-1);
    } else {
      idx_type xs = idx_type{1} << (qs[0] + 1);
      ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < qs_size; ++i) {
        ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs - 1);
        xs = idx_type{1} << (qs[i] + 1);
      }
      ms[qs_size] = ((idx_type{1} << num_qubits) - 1) ^ (xs - 1);
    }
  }

  template <typename QS>
  void GetXss(unsigned num_qubits, const QS& qs, unsigned qs_size,
              idx_type* xss) const {
    printf(": GetXss start\n");
    if (qs_size == 0) {
      xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
      }

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        xss[i] = a;
      }
    }
  }



// Original
  template <unsigned G, typename qs_type>
  void GetIndicesH(unsigned num_qubits, const qs_type& qs, unsigned qs_size,
                   IndicesH<G>& indices) const {
    printf(": GetIndicesH start\n");
    num_qubits = 40;
    if (qs_size == 0) {
      indices.ms[0] = idx_type(-1);
      indices.xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      indices.ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
        indices.ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs[i - 1] - 1);
      }
      indices.ms[g] = ((idx_type{1} << num_qubits) - 1) ^ (xs[g - 1] - 1);

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        indices.xss[i] = a;
      }
    }
  }

  template <unsigned G>
  void GetIndicesL(unsigned num_effective_qs, unsigned qmask,
                   IndicesL<G>& indices) const {
     printf(": GetIndicesL start\n");
    for (unsigned i = num_effective_qs + 1; i < (G + 1); ++i) {
      indices.ms[i] = 0;
    }

    for (unsigned i = (1 << num_effective_qs); i < indices.gsize; ++i) {
      indices.xss[i] = 0;
    }

    for (unsigned i = 0; i < indices.gsize; ++i) {
      indices.qis[i] = bits::ExpandBits(i, 5 + num_effective_qs, qmask);
    }

    unsigned tmask = ((1 << (5 + num_effective_qs)) - 1) ^ qmask;
    for (unsigned i = 0; i < 32; ++i) {
      indices.tis[i] = bits::ExpandBits(i, 5 + num_effective_qs, tmask);
    }
  }


  template <unsigned G>
  unsigned GetIndicesL(unsigned num_qubits, const std::vector<unsigned>& qs,
                       IndicesL<G>& indices) const {
                       
    num_qubits = 40;
      
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits);
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    if (qs.size() == num_low_qs) {
      while (ei < num_effective_qs && l++ < num_low_qs) {
        eqs[ei] = ei + 5;
        ++ei;
      }
    } else {
      while (ei < num_effective_qs && l < num_low_qs) {
        unsigned ei5 = ei + 5;
        eqs[ei] = ei5;
        if (qi < qs.size() && qs[qi] == ei5) {
          ++qi;
          qmaskh |= 1 << ei5;
        } else {
          ++l;
        }
        ++ei;
      }

      while (ei < num_effective_qs) {
        eqs[ei] = qs[qi++];
        qmaskh |= 1 << (ei + 5);
        ++ei;
      }
    }
    
    GetIndicesH(num_qubits, eqs, num_effective_qs, indices);
    
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    return num_effective_qs;
  }



template <unsigned G, typename qs_type>
void GetIndicesH_m(unsigned num_qubits, const qs_type& qs, unsigned qs_size,
                    IndicesH<G>& indices, size_t global_offset) {

    num_qubits = 40;
    
   if (qs_size == 0) {
      indices.ms[0] = idx_type(-1);
      indices.xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      indices.ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
        indices.ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs[i - 1] - 1);
      }
      indices.ms[g] = ((idx_type{1} << num_qubits) - 1) ^ (xs[g - 1] - 1);

        for (unsigned i = 0; i < gsize; ++i) {
            idx_type a = 0;
            for (unsigned k = 0; k < g; ++k) {
                a += xs[k] * ((i >> k) & 1);
            }
  
            indices.xss[i] = a;

        }
    }

}


template <unsigned G>
void GetIndicesL_m(unsigned num_effective_qs, unsigned qmask,
                    IndicesL<G>& indices, size_t global_offset) {
    unsigned num_qubits = 40;


      for (unsigned i = num_effective_qs + 1; i < (G + 1); ++i) {
      indices.ms[i] = 0;
    }

    for (unsigned i = (1 << num_effective_qs); i < indices.gsize; ++i) {
      indices.xss[i] = 0;
    }

    for (unsigned i = 0; i < indices.gsize; ++i) {
      indices.qis[i] = bits::ExpandBits(i, 5 + num_effective_qs, qmask);
    }

    unsigned tmask = ((1 << (5 + num_effective_qs)) - 1) ^ qmask;
    for (unsigned i = 0; i < 32; ++i) {
      indices.tis[i] = bits::ExpandBits(i, 5 + num_effective_qs, tmask);
    }


}



template <unsigned G>
unsigned GetIndicesL_m(unsigned num_qubits, const std::vector<unsigned>& qs,
                      IndicesL<G>& indices, size_t global_offset)  {
    num_qubits = 40;

    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
        qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits);
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    if (qs.size() == num_low_qs) {
        while (ei < num_effective_qs && l++ < num_low_qs) {
            eqs[ei] = ei + 5;
            ++ei;
        }
    } else {
        while (ei < num_effective_qs && l < num_low_qs) {
            unsigned ei5 = ei + 5;
            eqs[ei] = ei5;
            if (qi < qs.size() && qs[qi] == ei5) {
                ++qi;
                qmaskh |= 1 << ei5;
            } else {
                ++l;
            }
            ++ei;
        }

        while (ei < num_effective_qs) {
            eqs[ei] = qs[qi++];
            qmaskh |= 1 << (ei + 5);
            ++ei;
        }
    }


    
    GetIndicesH_m(num_qubits, eqs, num_effective_qs, indices, global_offset);
    GetIndicesL_m(num_effective_qs, qmaskh | qmaskl, indices, global_offset);

    return num_effective_qs;
}




  template <unsigned G>
  DataC GetIndicesLC(unsigned num_qubits, const std::vector<unsigned>& qs,
                     const std::vector<unsigned>& cqs, uint64_t cvals,
                     IndicesL<G>& indices) const {
                     
    printf(": GetIndicesLC start\n");
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ci = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    while (ai < num_qubits && l < num_low_qs) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        eqs[ei++] = ai;
        qmaskh |= 1 << (ai - ci);
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        eqs[ei++] = ai;
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = qi;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);

    return {cvalsh, num_aqs, num_effective_qs};
  }

  template <unsigned G>
  DataC GetIndicesLCL(unsigned num_qubits, const std::vector<unsigned>& qs,
                      const std::vector<unsigned>& cqs, uint64_t cvals,
                      IndicesLC<G>& indices) const {
    printf(": GetIndicesLCL start\n");
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;
    idx_type cmaskl = 0;
    idx_type cis_mask = 0;

    unsigned qi = 0;
    unsigned ci = 0;

    for (unsigned k = 0; k < 5; ++k) {
      if (qi < qs.size() && qs[qi] == k) {
        qmaskl |= 1 << (k - ci);
        ++qi;
      } else if (ci < cqs.size() && cqs[ci] == k) {
        cmaskl |= idx_type{1} << k;
        ++ci;
      }
    }

    unsigned num_low_qs = qi;
    unsigned num_low_cqs = ci;

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ei = 0;
    unsigned num_low = num_low_qs + num_low_cqs;
    unsigned remaining_low_cqs = num_low_cqs;
    unsigned effective_low_qs = num_low_qs;
    unsigned highest_cis_bit = 0;

    while (ai < num_qubits && l < num_low) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        if ((ai - ci) > 4) {
          eqs[ei++] = ai;
          qmaskh |= 1 << (ai - ci);
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          qmaskl |= 1 << (ai - ci);
          --remaining_low_cqs;
          ++effective_low_qs;
        }
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        if (remaining_low_cqs == 0) {
          eqs[ei++] = ai;
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          --remaining_low_cqs;
        }
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = effective_low_qs;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);
    idx_type cvalsl = bits::ExpandBits(idx_type(cvals), 5, cmaskl);

    cis_mask |= 31 ^ cmaskl;
    highest_cis_bit = highest_cis_bit < 5 ? 5 : highest_cis_bit;
    for (idx_type i = 0; i < 32; ++i) {
      auto c = bits::ExpandBits(i, highest_cis_bit + 1, cis_mask);
      indices.cis[i] = 2 * (c & 0xffffffe0) | (c & 0x1f) | cvalsl;
    }

    return {cvalsh, num_aqs, num_effective_qs, remaining_low_cqs};
  }



    void* AllocScratch(uint64_t size) const __attribute__((noinline)) {
    if (size > scratch_size_) {
            std::cout << "Allocating memory with cudaMalloc of size: 1" << size << std::endl;

      if (scratch_ != nullptr) {
              std::cout << "Allocating memory with cudaMalloc of size: 2" << size << std::endl;

        cudaFree(scratch_);
      }
              std::cout << "Allocating memory with cudaMalloc of size: 3" << size << std::endl;

      cudaMalloc(const_cast<void**>(&scratch_), size);
      printf(": cudaMalloc check size: %llu", size);
      
      const_cast<uint64_t&>(scratch_size_) = size;
      
    }

    
    return scratch_;
  }

  /* : 
d_ws: Device workspace pointer for storing gate and index data during CUDA operations.
h_ws (buf_size): Host-side workspace buffer with a maximum size, used to prepare data before copying to device.
h_ws: Pointer to host workspace, pointing to the start of h_ws0 buffer for easier handling.
scratch: Pointer to a dynamically allocated scratch buffer on the device, used for temporary storage.
scratch_size: Size of the scratch buffer in bytes, updated whenever more space is needed for computations.

*/


  
};


template <>
size_t SimulatorCUDA<float>::MAX_CACHE_SIZE = 10;         
template <>
size_t SimulatorCUDA<float>::MAX_STREAMS_PER_GPU = 8;
template <>
size_t SimulatorCUDA<float>::MAX_CPU_CACHE_SIZE = 64;

template <>
double SimulatorCUDA<float>::total_index_calc_time = 0.0;
template <>
double SimulatorCUDA<float>::total_apply_gate_time = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_H = 0.0;
template <>
float SimulatorCUDA<float>::total_gpu_time_H = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time = 0.0;
template <>
float SimulatorCUDA<float>::total_gpu_time = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_copy_h = 0.0;   
template <>
float SimulatorCUDA<float>::total_gpu_time_copy_h = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_copy_l = 0.0;   
template <>
float SimulatorCUDA<float>::total_gpu_time_copy_l = 0.0;
template <>
double SimulatorCUDA<float>::total_mpi_bcast_time = 0.0;

template<typename FP>
std::unordered_map<size_t, std::atomic<bool>>
    SimulatorCUDA<FP>::g_flush_in_progress;


}  // namespace qsim

#endif  // SIMULATOR_CUDA_H_

