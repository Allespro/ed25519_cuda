#ifndef ED25519_CUDA_CPU_SHIM
#define ED25519_CUDA_CPU_SHIM

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define __device__
#define __host__
#define __global__
#define __constant__

typedef int cudaError_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice
};
const cudaError_t cudaSuccess = 0;

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

cudaError_t cudaMalloc(void **ptr, size_t size);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);

inline int atomicAdd(int *address, int val) {
    int old = *address;
    *address = old + val;
    return old;
}

#endif
