#include "ed25519.cuh"
#include "sha512.cuh"
#include "ge.cuh"


__device__ void ed25519_kernel_create_keypair(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed) {
    ge_p3 A;

    sha512(seed, 32, private_key);
    private_key[0] &= 248;
    private_key[31] &= 63;
    private_key[31] |= 64;

    ge_scalarmult_base(&A, private_key);
    ge_p3_tobytes(public_key, &A);
}

__global__ void ed25519_kernel_create_keypair_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    ed25519_kernel_create_keypair(&public_key[compute_index * 32], &private_key[compute_index * 64], &seed[compute_index * 32]);
}
