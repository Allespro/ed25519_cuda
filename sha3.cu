#include "sha3.cuh"
#include "keccakf.cuh"

__device__ static void sha3_absorb_block(sha3_256_context *md, const unsigned char *block) {
    for (int i = 0; i < SHA3_256_RATE / 8; ++i) {
        uint64_t word = 0;
        for (int k = 0; k < 8; ++k) {
            word |= (uint64_t)block[i * 8 + k] << (8 * k);
        }
        md->state[i] ^= word;
    }
    keccakf(md->state);
}

__device__ int sha3_256_init(sha3_256_context *md) {
    if (md == NULL) {
        return 1;
    }
    for (int i = 0; i < 25; ++i) {
        md->state[i] = 0;
    }
    md->curlen = 0;
    return 0;
}

__device__ int sha3_256_update(sha3_256_context *md, const unsigned char *in, size_t inlen) {
    if (md == NULL) {
        return 1;
    }
    while (inlen > 0) {
        if (md->curlen == 0 && inlen >= SHA3_256_RATE) {
            sha3_absorb_block(md, in);
            in += SHA3_256_RATE;
            inlen -= SHA3_256_RATE;
        } else {
            size_t n = SHA3_256_RATE - md->curlen;
            if (n > inlen) {
                n = inlen;
            }
            for (size_t i = 0; i < n; ++i) {
                md->buf[md->curlen + i] = in[i];
            }
            md->curlen += n;
            in += n;
            inlen -= n;
            if (md->curlen == SHA3_256_RATE) {
                sha3_absorb_block(md, md->buf);
                md->curlen = 0;
            }
        }
    }
    return 0;
}

__device__ int sha3_256_final(sha3_256_context *md, unsigned char *out) {
    if (md == NULL || out == NULL) {
        return 1;
    }
    md->buf[md->curlen++] = 0x06;
    if (md->curlen == SHA3_256_RATE) {
        md->buf[SHA3_256_RATE - 1] |= 0x80;
        sha3_absorb_block(md, md->buf);
    } else {
        for (size_t i = md->curlen; i < SHA3_256_RATE - 1; ++i) {
            md->buf[i] = 0;
        }
        md->buf[SHA3_256_RATE - 1] = 0x80;
        sha3_absorb_block(md, md->buf);
    }
    md->curlen = 0;
    for (int i = 0; i < SHA3_256_HASH_LEN / 8; ++i) {
        for (int k = 0; k < 8; ++k) {
            out[i * 8 + k] = (unsigned char)(md->state[i] >> (8 * k));
        }
    }
    return 0;
}

__device__ int sha3_256(const unsigned char *message, size_t message_len, unsigned char *out) {
    sha3_256_context ctx;
    int ret;
    if ((ret = sha3_256_init(&ctx))) {
        return ret;
    }
    if ((ret = sha3_256_update(&ctx, message, message_len))) {
        return ret;
    }
    if ((ret = sha3_256_final(&ctx, out))) {
        return ret;
    }
    return 0;
}
