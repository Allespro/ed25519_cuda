#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <memory.h>
#include <assert.h>

unsigned int PERF_BLOCKS = 1024;
unsigned int PERF_THREADS = 256;

#include "src/ed25519.cuh"
#include "src/ge.cuh"
#include "src/sc.cuh"


void display_details(unsigned char *public_key_h, unsigned char *private_key_h) {
    printf("Logging the first public key and private key of the batch along with the signature for the input:\n\"9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a\" :\n\n");

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

int create_keypair(int enable_logging) {
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

    // Use a predefined seed to enbale reproducibility
    cudaMemcpy(seed, seed_h, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<1,1>>>(public_key, private_key, (const unsigned char*) seed, 1);

    if (enable_logging) {
        unsigned char public_key_h[32];
        unsigned char private_key_h[64];

        cudaMemcpy(public_key_h, public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(private_key_h, private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        //error = cudaGetLastError();
        //if (error != cudaSuccess) {
        //    printf("Error cuda ed25519 logging: %s \n", cudaGetErrorString(error));
        //}

        display_details(public_key_h, private_key_h);
    }

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_hf);
    return 1;
}

int main(int argc, char **argv) {
    int enable_logging = 0;
    for (int i = 0 ; i < argc; ++i) {
        if (strcmp(argv[i], "--logging") == 0 || strcmp(argv[i], "-l") == 0) {
            enable_logging = 1;
        }
    }

    printf(((enable_logging)? "\n\n---------- Test batch with single key pair ----------\n\n": ""));
    assert(create_keypair(enable_logging));
    return 0;
}
