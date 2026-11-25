#include "cuda_cpu_sim.h"

dim3 threadIdx(0, 0, 0);
dim3 blockIdx(0, 0, 0);
dim3 blockDim(1, 1, 1);
dim3 gridDim(1, 1, 1);

cudaError_t cudaMalloc(void **ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr ? cudaSuccess : (cudaError_t)1;
}

cudaError_t cudaFree(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t cudaFreeHost(void *ptr) {
    return cudaFree(ptr);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, size);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
    return cudaSuccess;
}
