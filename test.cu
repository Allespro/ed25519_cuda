#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <memory.h>
#include <assert.h>

#define PERF_BLOCKS 32768
#define PERF_THREADS 256

/* #define ED25519_DLL */
#define COLAB_ENVIRONMENT

#ifdef COLAB_ENVIRONMENT
/* 
 * In Google Colab we keep all files
 * under same directory for simplicity
 */
#include "ed25519.cuh"
#include "ge.cuh"
#include "sc.cuh"
#else
#include "src/ed25519.cuh"
#include "src/ge.cuh"
#include "src/sc.cuh"
#endif

void display_details(unsigned char *signature_h, unsigned char *public_key_h, unsigned char *private_key_h) {
    printf("Logging the first public key and private key of the batch along with the signature for the input:\n\"9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a\" :\n\n");
    
    printf("Signature\n");
    for (int i = 0; i < 64; ++i) {
        printf("%d  ", signature_h[i]);
    }
    printf("\n");

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

int test_batch_single_keypair(int enable_logging) {
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
    ed25519_kernel_create_seed(seed_hf);
    cudaMemcpy(seed, seed_hf, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Use a predefined seed to enbale reproducibility
    cudaMemcpy(seed, seed_h, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<1,1>>>(public_key, private_key, (const unsigned char*) seed, 1);

    unsigned char *messages;
    unsigned char *signature;
    int *verified;
    size_t *message_len;

    cudaMalloc(&messages, 1024 * 128 * sizeof(unsigned char));
    cudaMalloc(&signature, 1024 * 64 * sizeof(unsigned char));
    cudaMalloc(&message_len, 1025 * sizeof(size_t));
    cudaMalloc(&verified, 1024 * sizeof(int));

    unsigned char messages_h[1024 * 128 + 1];
    unsigned char out_h[1024 * 64];
    size_t message_len_h[1025];

    message_len_h[0] = 0;
    for (int i = 0; i < 1024; ++i) {
        message_len_h[i + 1] = message_len_h[i] + 128;
        strcpy((char *)&messages_h[i * 128],
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75"
            "a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    }
    memset(out_h, 0, 1024 * 64 * sizeof(unsigned char));

    cudaMemcpy(messages, messages_h, 1024 * 128 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(signature, out_h, 1024 * 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(message_len, message_len_h, 1025 * sizeof(size_t), cudaMemcpyHostToDevice);

    ed25519_kernel_sign_batch_single_keypair<<<8,128>>>(signature, messages, message_len, public_key, private_key, 1024);
    ed25519_kernel_verify_batch_single_keypair<<<8,128>>>(signature, messages, message_len, public_key, verified, 1024);


    int verified_h[1024];
    cudaMemcpy(verified_h, verified, 1024 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 sign-verify: %s \n", cudaGetErrorString(error));
	}

    if (enable_logging) {
        unsigned char signature_h[64];
        unsigned char public_key_h[32];
        unsigned char private_key_h[64];

        cudaMemcpy(signature_h, signature, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(public_key_h, public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(private_key_h, private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Error cuda ed25519 logging: %s \n", cudaGetErrorString(error));
        }

        display_details(signature_h, public_key_h, private_key_h);
    }


    cudaFree(messages);
    cudaFree(message_len);
    cudaFree(signature);
    cudaFree(verified);

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_hf);

    int flag = 1;
    for (int i = 0; i < 1024; ++i) {
        flag = flag & verified_h[i];
    }
    
    assert(flag);

    return 1;
}

int test_batch_multi_keypair_no_mapping(int enable_logging) {
    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed_h;
    unsigned char *seed;

    cudaMalloc(&public_key, 1024 * 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, 1024 * 64 * sizeof(unsigned char));
    cudaMalloc(&seed, 1024 * 32 * sizeof(unsigned char));
    cudaMallocHost(&seed_h, 1024 * 32 * sizeof(unsigned char));

    for (int i = 0; i < 1024; ++i) {
        ed25519_kernel_create_seed(&seed_h[i * 32]);
    }
    cudaMemcpy(seed, seed_h, 1024 * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<8,128>>>(public_key, private_key, (const unsigned char*) seed, 1024);

    unsigned char *messages;
    unsigned char *signature;
    int *verified;
    size_t *message_len;

    cudaMalloc(&messages, 1024 * 128 * sizeof(unsigned char));
    cudaMalloc(&signature, 1024 * 64 * sizeof(unsigned char));
    cudaMalloc(&message_len, 1025 * sizeof(size_t));
    cudaMalloc(&verified, 1024 * sizeof(int));

    unsigned char messages_h[1024 * 128 + 1];
    unsigned char out_h[1024 * 64];
    size_t message_len_h[1025];

    message_len_h[0] = 0;
    for (int i = 0; i < 1024; ++i) {
        message_len_h[i + 1] = message_len_h[i] + 128;
        strcpy((char *)&messages_h[i * 128],
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75"
            "a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    }
    memset(out_h, 0, 1024 * 64 * sizeof(unsigned char));

    cudaMemcpy(messages, messages_h, 1024 * 128 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(signature, out_h, 1024 * 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(message_len, message_len_h, 1025 * sizeof(size_t), cudaMemcpyHostToDevice);

    ed25519_kernel_sign_batch_multi_keypair<<<8,128>>>(signature, messages, message_len, public_key, private_key, NULL, 1024);
    ed25519_kernel_verify_batch_multi_keypair<<<8,128>>>(signature, messages, message_len, public_key, verified, NULL, 1024);


    int verified_h[1024];
    cudaMemcpy(verified_h, verified, 1024 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 sign-verify: %s \n", cudaGetErrorString(error));
	}

    if (enable_logging) {
        unsigned char signature_h[64];
        unsigned char public_key_h[32];
        unsigned char private_key_h[64];

        cudaMemcpy(signature_h, signature, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(public_key_h, public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(private_key_h, private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Error cuda ed25519 logging: %s \n", cudaGetErrorString(error));
        }

        display_details(signature_h, public_key_h, private_key_h);
    }


    cudaFree(messages);
    cudaFree(message_len);
    cudaFree(signature);
    cudaFree(verified);

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_h);

    int flag = 1;
    for (int i = 0; i < 1024; ++i) {
        flag = flag & verified_h[i];
    }
    
    assert(flag);

    return 1;
}

int test_batch_multi_keypair_with_mapping(int enable_logging) {
    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed;
    int *key_pair_mapping;

    unsigned char seed_h[33] = "01234567890123456789012345678901";
    int key_pair_mapping_h[1024];

    cudaMalloc(&public_key, 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, 64 * sizeof(unsigned char));
    cudaMalloc(&seed, 1024 * 32 * sizeof(unsigned char));
    cudaMalloc(&key_pair_mapping, 1024 * sizeof(int));

    memset(key_pair_mapping_h, 0, 1024 * sizeof(int));

    cudaMemcpy(seed, seed_h, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(key_pair_mapping, key_pair_mapping_h, 1024 * sizeof(int), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<1,1>>>(public_key, private_key, (const unsigned char*) seed, 1);

    unsigned char *messages;
    unsigned char *signature;
    int *verified;
    size_t *message_len;

    cudaMalloc(&messages, 1024 * 128 * sizeof(unsigned char));
    cudaMalloc(&signature, 1024 * 64 * sizeof(unsigned char));
    cudaMalloc(&message_len, 1025 * sizeof(size_t));
    cudaMalloc(&verified, 1024 * sizeof(int));

    unsigned char messages_h[1024 * 128 + 1];
    unsigned char out_h[1024 * 64];
    size_t message_len_h[1025];

    message_len_h[0] = 0;
    for (int i = 0; i < 1024; ++i) {
        message_len_h[i + 1] = message_len_h[i] + 128;
        strcpy((char *)&messages_h[i * 128],
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75"
            "a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    }
    memset(out_h, 0, 1024 * 64 * sizeof(unsigned char));

    cudaMemcpy(messages, messages_h, 1024 * 128 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(signature, out_h, 1024 * 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(message_len, message_len_h, 1025 * sizeof(size_t), cudaMemcpyHostToDevice);

    ed25519_kernel_sign_batch_multi_keypair<<<8,128>>>(signature, messages, message_len, public_key, private_key, key_pair_mapping, 1024);
    ed25519_kernel_verify_batch_multi_keypair<<<8,128>>>(signature, messages, message_len, public_key, verified, key_pair_mapping, 1024);


    int verified_h[1024];
    cudaMemcpy(verified_h, verified, 1024 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 sign-verify: %s \n", cudaGetErrorString(error));
	}

    if (enable_logging) {
        unsigned char signature_h[64];
        unsigned char public_key_h[32];
        unsigned char private_key_h[64];

        cudaMemcpy(signature_h, signature, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(public_key_h, public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(private_key_h, private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Error cuda ed25519 logging: %s \n", cudaGetErrorString(error));
        }

        display_details(signature_h, public_key_h, private_key_h);
    }


    cudaFree(messages);
    cudaFree(message_len);
    cudaFree(signature);
    cudaFree(verified);

    cudaFree(key_pair_mapping);
    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);

    int flag = 1;
    for (int i = 0; i < 1024; ++i) {
        flag = flag & verified_h[i];
    }
    
    assert(flag);

    return 1;
}

int test_batch_add_scalar(int enable_logging) {
    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed_h;
    unsigned char *seed;

    unsigned char *scalar;
    unsigned char scalar_h[1024 * 33];

    unsigned char public_key_h[3][33];
    unsigned char private_key_h[3][65];

    int *scalar_mapping;
    int scalar_mapping_h[1024];

    cudaMalloc(&public_key, 1024 * 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, 1024 * 64 * sizeof(unsigned char));
    cudaMalloc(&seed, 1024 * 32 * sizeof(unsigned char));
    cudaMalloc(&scalar, 1024 * 32 * sizeof(unsigned char));
    cudaMalloc(&scalar_mapping, 1024 * sizeof(int));
    cudaMallocHost(&seed_h, 1024 * 32 * sizeof(unsigned char));

    for (int i = 0; i < 1024; ++i) {
        strcpy((char*) &scalar_h[i * 32], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        ed25519_kernel_create_seed(&seed_h[i * 32]);
    }

    memset(scalar_mapping_h, 0, 1024 * sizeof(int));

    cudaMemcpy(seed, seed_h, 1024 * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(scalar, scalar_h, 1024 * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(scalar_mapping, scalar_mapping_h, 1024 * sizeof(int), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<8,128>>>(public_key, private_key, (const unsigned char*) seed, 1024);
    ed25519_kernel_add_scalar_batch<<<8,128>>>(public_key, private_key, scalar, 1024);
    cudaMemcpy(public_key_h[0], public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(private_key_h[0], private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    ed25519_kernel_create_keypair_batch<<<8,128>>>(public_key, private_key, (const unsigned char*) seed, 1024);
    ed25519_kernel_add_multi_scalar_batch<<<8,128>>>(public_key, private_key, scalar, NULL, 1024);
    cudaMemcpy(public_key_h[1], public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(private_key_h[1], private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    ed25519_kernel_create_keypair_batch<<<8,128>>>(public_key, private_key, (const unsigned char*) seed, 1024);
    ed25519_kernel_add_multi_scalar_batch<<<8,128>>>(public_key, private_key, scalar, scalar_mapping, 1024);
    cudaMemcpy(public_key_h[2], public_key, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(private_key_h[2], private_key, 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 add scalar: %s \n", cudaGetErrorString(error));
	}

    public_key_h[0][32] = '\0';
    public_key_h[1][32] = '\0';
    public_key_h[2][32] = '\0';

    private_key_h[0][64] = '\0';
    private_key_h[1][64] = '\0';
    private_key_h[2][64] = '\0';

    int flag = strcmp((char*) public_key_h[0], (char*) public_key_h[1]) || strcmp((char*) public_key_h[0], (char*) public_key_h[2]);
    flag = flag || strcmp((char*) private_key_h[0], (char*) private_key_h[1]) || strcmp((char*) private_key_h[0], (char*) private_key_h[2]);

    if (enable_logging) {
        display_key_pair(public_key_h, private_key_h, 3);
    }

    cudaFree(scalar);
    cudaFree(scalar_mapping);

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_h);

    assert(flag == 0);

    return 1;
}

int test_key_exchange(int enable_logging) {
    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed_h;
    unsigned char *seed;
    unsigned char *shared_secret_1;
    unsigned char *shared_secret_2;

    unsigned char shared_secret_h[2][33];

    cudaMalloc(&public_key, 2 * 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, 2 * 64 * sizeof(unsigned char));
    cudaMalloc(&seed, 2 * 32 * sizeof(unsigned char));
    cudaMalloc(&shared_secret_1, 32 * sizeof(unsigned char));
    cudaMalloc(&shared_secret_2, 32 * sizeof(unsigned char));
    cudaMallocHost(&seed_h, 2 * 32 * sizeof(unsigned char));

    for (int i = 0; i < 2; ++i) {
        ed25519_kernel_create_seed(&seed_h[i * 32]);
    }

    cudaMemcpy(seed, seed_h, 2 * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    ed25519_kernel_create_keypair_batch<<<1,2>>>(public_key, private_key, (const unsigned char*) seed, 2);
    ed25519_kernel_key_exchange_batch<<<1,1>>>(shared_secret_1, public_key, &private_key[64], 1);
    ed25519_kernel_key_exchange_batch<<<1,1>>>(shared_secret_2, &public_key[32], private_key, 1);

    cudaMemcpy(shared_secret_h[0], shared_secret_1, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(shared_secret_h[1], shared_secret_2, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 add scalar: %s \n", cudaGetErrorString(error));
	}

    shared_secret_h[0][32] = '\0';
    shared_secret_h[1][32] = '\0';

    if (enable_logging) {
        display_secret(shared_secret_h);
    }

    int flag = strcmp((char*) shared_secret_h[0], (char*) shared_secret_h[1]);

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFreeHost(seed_h);

    assert(flag == 0);

    return 1;
}

void test_performance(int enable_logging) {
    clock_t start;
    clock_t end;

    unsigned char *public_key;
    unsigned char *private_key;
    unsigned char *seed_h;
    unsigned char *seed;
    unsigned char *signature;
    int *verified;

    unsigned char *messages;
    unsigned char *messages_h = (unsigned char *) malloc((PERF_BLOCKS * PERF_THREADS * 128 + 1) * sizeof(unsigned char));

    size_t *message_len;
    size_t *message_len_h = (size_t *) malloc((PERF_BLOCKS * PERF_THREADS + 1) * sizeof(size_t));

    unsigned char *scalar;
    unsigned char *scalar_h = (unsigned char *) malloc((PERF_BLOCKS * PERF_THREADS * 32 + 1) * sizeof(unsigned char));

    unsigned char *shared;

    cudaMalloc(&public_key, (PERF_BLOCKS * PERF_THREADS + 1) * 32 * sizeof(unsigned char));
    cudaMalloc(&private_key, (PERF_BLOCKS * PERF_THREADS + 1) * 64 * sizeof(unsigned char));
    cudaMalloc(&seed, (PERF_BLOCKS * PERF_THREADS + 1) * 32 * sizeof(unsigned char));
    cudaMalloc(&messages, PERF_BLOCKS * PERF_THREADS * 128 * sizeof(unsigned char));
    cudaMalloc(&message_len, (PERF_BLOCKS * PERF_THREADS + 1) * sizeof(size_t));
    cudaMalloc(&signature, PERF_BLOCKS * PERF_THREADS * 64 * sizeof(unsigned char));
    cudaMalloc(&scalar, PERF_BLOCKS * PERF_THREADS * 32 * sizeof(unsigned char));
    cudaMalloc(&shared, PERF_BLOCKS * PERF_THREADS * 32 * sizeof(unsigned char));
    cudaMalloc(&verified, PERF_BLOCKS * PERF_THREADS * sizeof(int));
    cudaMallocHost(&seed_h, (PERF_BLOCKS * PERF_THREADS + 1) * 32 * sizeof(unsigned char));
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - Memory allocation: %s \n", cudaGetErrorString(error));
	}

    message_len_h[0] = 0;
    for (int i = 0; i < PERF_BLOCKS * PERF_THREADS; ++i) {
        message_len_h[i + 1] = message_len_h[i] + 32;
        ed25519_kernel_create_seed(&seed_h[i * 32]);
        strcpy((char *)&messages_h[i * 128],
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60d75"
            "a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
        strcpy((char*) &scalar_h[i * 32], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    }
    ed25519_kernel_create_seed(&seed_h[PERF_BLOCKS * PERF_THREADS * 32]);

    cudaMemcpy(seed, seed_h, (PERF_BLOCKS * PERF_THREADS + 1) * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(messages, messages_h, PERF_BLOCKS * PERF_THREADS * 128 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(message_len, message_len_h, (PERF_BLOCKS * PERF_THREADS + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(scalar, scalar_h, PERF_BLOCKS * PERF_THREADS * 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_create_keypair_batch<<<PERF_BLOCKS + 1,PERF_THREADS>>>(public_key, private_key, (const unsigned char*) seed, PERF_BLOCKS * PERF_THREADS + 1);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - keypair generation: %s \n", cudaGetErrorString(error));
	}

    printf("Key pair generation performance: %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS + 1) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_sign_batch_single_keypair<<<PERF_BLOCKS,PERF_THREADS>>>(signature, messages, message_len, public_key, private_key, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - sign (single key pair): %s \n", cudaGetErrorString(error));
	}

    printf("Sign performance (single key pair): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_verify_batch_single_keypair<<<PERF_BLOCKS,PERF_THREADS>>>(signature, messages, message_len, public_key, verified, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - verify (single key pair): %s \n", cudaGetErrorString(error));
	}

    printf("Verify performance (single key pair): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_sign_batch_multi_keypair<<<PERF_BLOCKS,PERF_THREADS>>>(signature, messages, message_len, public_key, private_key, NULL, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - sign (multiple key pair): %s \n", cudaGetErrorString(error));
	}

    printf("Sign performance (multiple key pair): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_verify_batch_multi_keypair<<<PERF_BLOCKS,PERF_THREADS>>>(signature, messages, message_len, public_key, verified, NULL, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - verify (multiple key pair): %s \n", cudaGetErrorString(error));
	}

    printf("Verify performance (multiple key pair): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_add_scalar_batch<<<PERF_BLOCKS,PERF_THREADS>>>(public_key, private_key, scalar, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - add scalar (same): %s \n", cudaGetErrorString(error));
	}

    printf("Add scalar performance (same scalar): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_add_multi_scalar_batch<<<PERF_BLOCKS,PERF_THREADS>>>(public_key, private_key, scalar, NULL, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - add scalar (multiple): %s \n", cudaGetErrorString(error));
	}

    printf("Add scalar performance (multiple scalar): %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    cudaDeviceSynchronize();
    start = clock();

    ed25519_kernel_key_exchange_batch<<<PERF_BLOCKS,PERF_THREADS>>>(shared, &public_key[32], private_key, PERF_BLOCKS * PERF_THREADS);

    cudaDeviceSynchronize();
    end = clock();

    error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda ed25519 performance test - key exchange: %s \n", cudaGetErrorString(error));
	}

    printf("Key Exchange performance: %fus per key pair\n", ((double) ((end - start) * 1000)) / CLOCKS_PER_SEC / (PERF_BLOCKS * PERF_THREADS) * 1000);

    free(messages_h);
    free(message_len_h);
    free(scalar_h);

    cudaFree(public_key);
    cudaFree(private_key);
    cudaFree(seed);
    cudaFree(messages);
    cudaFree(message_len);
    cudaFree(signature);
    cudaFree(scalar);
    cudaFree(shared);
    cudaFree(verified);
    cudaFreeHost(seed_h);
}

int main(int argc, char **argv) {
    int enable_logging = 0;
    int enable_perf_logging = 0;
    for (int i = 0 ; i < argc; ++i) {
        if (strcmp(argv[i], "--perf") == 0 || strcmp(argv[i], "-p") == 0) {
            enable_perf_logging = 1;
        } else if (strcmp(argv[i], "--logging") == 0 || strcmp(argv[i], "-l") == 0) {
            enable_logging = 1;
        }
    }

    printf(((enable_logging)? "\n\n---------- Test batch with single key pair ----------\n\n": ""));
    assert(test_batch_single_keypair(enable_logging));

    printf(((enable_logging)? "\n\n---------- Test batch with multiple key pair - 1024 unique key pair ----------\n\n": ""));
    assert(test_batch_multi_keypair_no_mapping(enable_logging));

    printf(((enable_logging)? "\n\n---------- Test batch with multiple key pair - With key-pair mapping ----------\n\n": ""));
    assert(test_batch_multi_keypair_with_mapping(enable_logging));

    printf(((enable_logging)? "\n\n---------- Test batch add scalar ----------\n\n": ""));
    assert(test_batch_add_scalar(enable_logging));

    printf(((enable_logging)? "\n\n---------- Test batch key exchange ----------\n\n": ""));
    assert(test_key_exchange(enable_logging));

    if (enable_perf_logging) {
        printf(((enable_logging)? "\n\n---------- Performance ----------\n\n": ""));
        test_performance(enable_logging);
    }

    return 0;
}
