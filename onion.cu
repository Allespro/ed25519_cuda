#include "keccakf.cuh"



__global__ void onion_address(unsigned char *public_key) {
    uint8_t checksum[200] = ".onion checksum";
    for (int i = 0; i < 32; ++i) checksum[15 + i] = public_key[i];
    checksum[47] = 0x03;
    // pad checksum up r bits
    checksum[48] = 0x01; // pad10*1 start
    checksum[135] = 0x80; // pad10*1 end

    // cast checksum to S for keccakf implementation (blocks of 64 bits)
    uint64_t *S[25];
    for (size_t i = 0; i < 25; i++) {
        S[i] = (uint64_t *)&checksum[i * 8];
    }

    // run the keccakf transformation
    keccakf(*S); 
}