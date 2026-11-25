#include "onion.cuh"
#include "sha3.cuh"

__device__ void onion_checksum(const unsigned char *public_key, unsigned char *checksum) {
    unsigned char input[48];
    unsigned char digest[32];
    static const char prefix[] = ".onion checksum";

    for (int i = 0; i < 15; ++i) {
        input[i] = (unsigned char)prefix[i];
    }
    for (int i = 0; i < ONION_PUBLIC_KEY_SIZE; ++i) {
        input[15 + i] = public_key[i];
    }
    input[47] = ONION_VERSION_BYTE;

    sha3_256(input, 48, digest);
    checksum[0] = digest[0];
    checksum[1] = digest[1];
}

__device__ void onion_encode_address(const unsigned char *public_key, const unsigned char *checksum, char *address) {
    unsigned char bytes[ONION_KEY_BYTES];
    int bit_buffer = 0;
    int bit_count = 0;
    int out = 0;

    for (int i = 0; i < ONION_PUBLIC_KEY_SIZE; ++i) {
        bytes[i] = public_key[i];
    }
    for (int i = 0; i < ONION_CHECKSUM_SIZE; ++i) {
        bytes[ONION_PUBLIC_KEY_SIZE + i] = checksum[i];
    }
    bytes[ONION_KEY_BYTES - 1] = ONION_VERSION_BYTE;

    for (int i = 0; i < ONION_KEY_BYTES; ++i) {
        bit_buffer = (bit_buffer << 8) | bytes[i];
        bit_count += 8;
        while (bit_count >= 5) {
            address[out++] = ONION_BASE32_ALPHABET[(bit_buffer >> (bit_count - 5)) & 31];
            bit_count -= 5;
        }
    }
}

__global__ void onion_address(const unsigned char *public_key, char *address, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    const unsigned char *key = public_key + compute_index * ONION_PUBLIC_KEY_SIZE;
    char *out = address + compute_index * (ONION_ADDRESS_LEN + 1);
    unsigned char checksum[ONION_CHECKSUM_SIZE];

    onion_checksum(key, checksum);
    onion_encode_address(key, checksum, out);
    out[ONION_ADDRESS_LEN] = '\0';
}

__device__ int onion_pattern_matches(const char *address, const char *prefix, const char *suffix) {
    int prefix_len = 0;
    int suffix_len = 0;
    int i;

    if (prefix == NULL) {
        prefix = "";
    }
    if (suffix == NULL) {
        suffix = "";
    }
    while (prefix[prefix_len] != '\0') {
        ++prefix_len;
    }
    while (suffix[suffix_len] != '\0') {
        ++suffix_len;
    }
    if (prefix_len + suffix_len > ONION_ADDRESS_LEN) {
        return 0;
    }
    for (i = 0; i < prefix_len; ++i) {
        if (address[i] != prefix[i]) {
            return 0;
        }
    }
    for (i = 0; i < suffix_len; ++i) {
        if (address[ONION_ADDRESS_LEN - suffix_len + i] != suffix[i]) {
            return 0;
        }
    }
    return 1;
}

__global__ void onion_search(const unsigned char *public_key, const char *prefix, const char *suffix, int *found, int limit) {
    int compute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (compute_index >= limit) {
        return;
    }

    const unsigned char *key = public_key + compute_index * ONION_PUBLIC_KEY_SIZE;
    char address[ONION_ADDRESS_LEN + 1];
    unsigned char checksum[ONION_CHECKSUM_SIZE];

    onion_checksum(key, checksum);
    onion_encode_address(key, checksum, address);
    address[ONION_ADDRESS_LEN] = '\0';
    found[compute_index] = onion_pattern_matches(address, prefix, suffix);
}
