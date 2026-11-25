#ifndef ED25519_TEST_LAUNCH
#define ED25519_TEST_LAUNCH

#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__

#include <cuda_runtime.h>

template <typename F, typename... Args>
void launch(F kernel, unsigned int grid, unsigned int block, Args... args) {
    kernel<<<grid, block>>>(args...);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

#else

#include "cpu_sim/cuda_cpu_sim.h"

template <typename F, typename... Args>
void launch(F kernel, unsigned int grid, unsigned int block, Args... args) {
    gridDim = dim3(grid, 1, 1);
    blockDim = dim3(block, 1, 1);
    for (unsigned int b = 0; b < grid; ++b) {
        blockIdx = dim3(b, 0, 0);
        for (unsigned int t = 0; t < block; ++t) {
            threadIdx = dim3(t, 0, 0);
            kernel(args...);
        }
    }
}

#endif

#endif
