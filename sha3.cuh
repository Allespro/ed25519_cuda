#ifndef SHA3_256_H
#define SHA3_256_H

#include <stddef.h>

#include "fixedint.cuh"

#define SHA3_256_RATE 136
#define SHA3_256_HASH_LEN 32

typedef struct sha3_256_context_ {
    uint64_t state[25];
    size_t curlen;
    unsigned char buf[SHA3_256_RATE];
} sha3_256_context;

__device__ int sha3_256_init(sha3_256_context *md);
__device__ int sha3_256_update(sha3_256_context *md, const unsigned char *in, size_t inlen);
__device__ int sha3_256_final(sha3_256_context *md, unsigned char *out);
__device__ int sha3_256(const unsigned char *message, size_t message_len, unsigned char *out);

#endif
