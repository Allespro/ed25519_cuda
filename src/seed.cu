#include "ed25519.cuh"
#include <stdio.h>
#include <assert.h>

__host__ void ed25519_kernel_create_seed(unsigned char *seed) {
    FILE *f = fopen("/dev/urandom", "rb");

    if (f == NULL) {
        assert(0);
    }

    fread(seed, 1, 32, f);
    fclose(f);
}
