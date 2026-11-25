#ifndef ONION
#define ONION

#include <stddef.h>

#define ONION_ADDRESS_LEN 56
#define ONION_PUBLIC_KEY_SIZE 32
#define ONION_CHECKSUM_SIZE 2
#define ONION_VERSION_BYTE 0x03
#define ONION_KEY_BYTES 35
#define ONION_BASE32_ALPHABET "abcdefghijklmnopqrstuvwxyz234567"

__global__ void onion_address(const unsigned char *public_key, char *address, int limit);
__global__ void onion_search(const unsigned char *public_key, const char *prefix, const char *suffix, int *found, int limit);

#endif
