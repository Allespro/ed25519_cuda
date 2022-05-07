#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <memory.h>
#include <assert.h>

unsigned int PERF_BLOCKS = 1024;
unsigned int PERF_THREADS = 256;

#include "ed25519.cuh"
#include "ge.cuh"
#include "sc.cuh"

void pubkey(unsigned char *seed, unsigned char *pk)
{
  uint8_t h[64];
  ge_p3 A;
  sha512(seed,32,h);
  h[0] &= 248;
  h[31] &= 63;
  h[31] |= 64;
  //ge_scalarmult_base(&A,h);
  //ge_p3_tobytes(pk,&A);

}

void display_details(unsigned char *public_key_h, unsigned char *private_key_h) {
    printf("Public Key\n");
    for (int i = 0; i < 32; ++i) {
        printf("%d  ", public_key_h[i]);
    }
    printf("\n");

    printf("Private Key\n");
    for (int i = 0; i < 64; ++i) {
        printf("%d  ", private_key_h[i]);
    }
    printf("\n");
    onion_address(public_key_h);
}

void display_key_pair(unsigned char public_keys[][33], unsigned char private_keys[][65], int n) {
    printf("Logging the key pairs after scalar addition of \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" :\n\n");

    for (int j = 0; j < n; ++j) {
        printf("Public Key %d\n", j + 1);
        for (int i = 0; i < 32; ++i) {
            printf("%d  ", public_keys[j][i]);
        }
        printf("\n");

        printf("Private Key %d\n", j + 1);
        for (int i = 0; i < 64; ++i) {
            printf("%d  ", private_keys[j][i]);
        }
        printf("\n\n");
    }
}

void display_secret(unsigned char secret[2][33]) {
    printf("Following are the secrets for the key exchange (both secret should be same):\n\n");

    for(int j = 0; j < 2; ++j) {
        printf("Secret %d:\n", j + 1);
        for (int i = 0; i < 32; ++i) {
            printf("%d  ", secret[j][i]);
        }
        printf("\n");
    }

}

int create_keypair(bool enable_logging, bool test_seed) {
    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed_hf;
    unsigned char *seed;
    unsigned char seed_h[33] = "01234567890123456789012345678901";

    cudaMalloc(&public_key, 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, 64 * sizeof(unsigned char));
    cudaMalloc(&seed, 32 * sizeof(unsigned char));
    cudaMallocHost(&seed_hf, 32 * sizeof(unsigned char));

    // Test whether create seed __host__ function works
    ed25519_kernel_create_seed(seed_hf,1);
    cudaMemcpy(seed, seed_hf, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (test_seed) {
        cudaMemcpy(seed, seed_h, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    ed25519_kernel_create_keypair_batch<<<1,1>>>(public_key, private_key, (const unsigned char*) seed, 1);

    if (enable_logging) {
        unsigned char public_key_h[32];
        unsigned char private_key_h[64];
        cudaMemcpy(public_key_h, public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(private_key_h, private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        display_details(public_key_h, private_key_h);
    }

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_hf);
    return 1;
}

int main(int argc, char **argv) {
    bool enable_logging = false;
    bool test_seed = false;
    for (int i = 0 ; i < argc; ++i) {
        if (strcmp(argv[i], "--logging") == 0 || strcmp(argv[i], "-l") == 0) {
            enable_logging = true;
        } else if (strcmp(argv[i], "--testseed") == 0 || strcmp(argv[i], "-t") == 0) {
            test_seed = true;
        }
    }

    printf(((enable_logging)? "\n\n---------- Test batch with single key pair ----------\n\n": ""));
    assert(create_keypair(enable_logging, test_seed));
    return 0;
}
