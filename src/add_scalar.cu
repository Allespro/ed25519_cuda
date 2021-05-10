#include "ed25519.cuh"
#include "ge.cuh"
#include "sc.cuh"
#include "sha512.cuh"
#include "util.cuh"


/* see http://crypto.stackexchange.com/a/6215/4697 */
__device__ void ed25519_kernel_add_scalar(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar) {
    const unsigned char SC_1[32] = {1}; /* scalar with value 1 */
    
    unsigned char *n; 
    ge_p3 nB;
    ge_p1p1 A_p1p1;
    ge_p3 A;
    ge_p3 public_key_unpacked;
    ge_cached T;

    sha512_context hash;
    unsigned char *hashbuf;

    cudaMalloc(&n, 32 * sizeof(unsigned char));
    cudaMalloc(&hashbuf, 64 * sizeof(unsigned char));
    
    kernel_add_scalar_copy_scalar<<<1,32>>>(n, scalar);
    cudaDeviceSynchronize();

    n[31] = scalar[31] & 127;

    /* private key: a = n + t */
    if (private_key) {
        sc_muladd(private_key, SC_1, n, private_key);

        // https://github.com/orlp/ed25519/issues/3
        sha512_init(&hash);
        sha512_update(&hash, private_key + 32, 32);
        sha512_update(&hash, scalar, 32);
        sha512_final(&hash, hashbuf);

        kernel_add_scalar_copy_pk<<<1,32>>>(private_key, hashbuf);
        cudaDeviceSynchronize();
    }

    /* public key: A = nB + T */
    if (public_key) {
        /* if we know the private key we don't need a point addition, which is faster */
        /* using a "timing attack" you could find out wether or not we know the private
           key, but this information seems rather useless - if this is important pass
           public_key and private_key seperately in 2 function calls */
        if (private_key) {
            ge_scalarmult_base(&A, private_key);
        } else {
            /* unpack public key into T */
            ge_frombytes_negate_vartime(&public_key_unpacked, public_key);
            fe_neg(public_key_unpacked.X, public_key_unpacked.X); /* undo negate */
            fe_neg(public_key_unpacked.T, public_key_unpacked.T); /* undo negate */
            ge_p3_to_cached(&T, &public_key_unpacked);

            /* calculate n*B */
            ge_scalarmult_base(&nB, n);

            /* A = n*B + T */
            ge_add(&A_p1p1, &nB, &T);
            ge_p1p1_to_p3(&A, &A_p1p1);
        }
            
        /* pack public key */
        ge_p3_tobytes(public_key, &A);
    }

    cudaFree(n);
    cudaFree(hashbuf);
    cudaDeviceSynchronize();
}

__global__ void ed25519_kernel_add_scalar_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    ed25519_kernel_add_scalar(&public_key[compute_index * 32], &private_key[compute_index * 64], scalar);
}

__global__ void ed25519_kernel_add_multi_scalar_batch(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar, int *scalar_mapping, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    int scalar_index = ((scalar_mapping == NULL)? compute_index: scalar_mapping[compute_index]);
    ed25519_kernel_add_scalar(&public_key[compute_index * 32], &private_key[compute_index * 64], &scalar[scalar_index * 32]);
}
