#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
const int max_grid_blocks = 512;

#define CUDA_CHECK(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error ") + std::to_string(__LINE__) + \
                ": " + cudaGetErrorString(e)); \
        } \
    } while(0)

inline int get_cuda_device_count() // get number of CUDA devices available
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
}

inline void select_best_cuda_device()
{
    int count = get_cuda_device_count();
    if (count == 0) throw std::runtime_error("No CUDA device found");
    int best = 0;
    int max_mp = 0;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (prop.multiProcessorCount > max_mp) {
            max_mp = prop.multiProcessorCount;
            best = i;
        }
    }
    CUDA_CHECK(cudaSetDevice(best));
}

inline size_t get_free_gpu_memory()
{
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

inline void* safe_cuda_malloc(size_t bytes, float reserve_ratio = 0.1f)
{
    size_t free_mem = get_free_gpu_memory();
    size_t limit = (size_t)(free_mem * (1.0f - reserve_ratio));
    if (bytes > limit) {
        throw std::runtime_error(
            "Requested " + std::to_string(bytes / 1024 / 1024) +
            " MB but only " + std::to_string(limit / 1024 / 1024) +
            " MB available after " + std::to_string((int)(reserve_ratio * 100)) + "% reserve"
        );
    }
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}