#include "ed25519.cuh"
#include "sha512.cuh"
#include "ge.cuh"
#include "sc.cuh"


__device__ void ed25519_kernel_sign(unsigned char *signature, const unsigned char *message, size_t message_len, const unsigned char *public_key, const unsigned char *private_key) {
    sha512_context hash;
    unsigned char hram[64];
    unsigned char r[64];
    ge_p3 R;


    sha512_init(&hash);
    sha512_update(&hash, private_key + 32, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, r);

    sc_reduce(r);
    ge_scalarmult_base(&R, r);
    ge_p3_tobytes(signature, &R);

    sha512_init(&hash);
    sha512_update(&hash, signature, 32);
    sha512_update(&hash, public_key, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, hram);

    sc_reduce(hram);
    sc_muladd(signature + 32, hram, private_key, r);
}

__global__ void ed25519_kernel_sign_batch_single_keypair(unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, const unsigned char *private_key, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    ed25519_kernel_sign(&signature[compute_index * 64], &message[message_len[compute_index]], message_len[compute_index + 1] - message_len[compute_index], public_key, private_key);
}

__global__ void ed25519_kernel_sign_batch_multi_keypair(unsigned char *signature, const unsigned char *message, size_t *message_len, const unsigned char *public_key, const unsigned char *private_key, int *key_mapping, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    int key_index = ((key_mapping == NULL)? compute_index: key_mapping[compute_index]);
    ed25519_kernel_sign(&signature[compute_index * 64], &message[message_len[compute_index]], message_len[compute_index + 1] - message_len[compute_index], &public_key[key_index * 32], &private_key[key_index * 64]);
}
