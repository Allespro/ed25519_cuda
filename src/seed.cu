#include "ed25519.cuh"
#include <stdio.h>
#include <assert.h>

__host__ void ed25519_kernel_create_seed(unsigned char *seed, int batch_size) {
    FILE *f = fopen("/dev/urandom", "rb");

    if (f == NULL) {
        assert(0);
    }

    (void) fread(seed, 1, 32 * batch_size, f);
    fclose(f);
}
