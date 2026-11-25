#ifndef ED25519_TEST_COMMON_H
#define ED25519_TEST_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "launch.h"
#include "cpu_sim/cuda_cpu_sim.h"

#include "ed25519.cuh"
#include "fe.cuh"
#include "ge.cuh"
#include "sc.cuh"
#include "sha512.cuh"
#include "keccakf.cuh"
#include "sha3.cuh"
#include "onion.cuh"

#include "test_vectors.h"

static int test_failures = 0;
static int test_checks = 0;

static std::string to_hex(const unsigned char *p, size_t n) {
    static const char *digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 15];
    }
    return s;
}

static void unhex(const char *hex, unsigned char *out, size_t n) {
    if (strlen(hex) != n * 2) {
        fprintf(stderr, "bad test vector length\n");
        exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = (unsigned char)strtoul(std::string(hex + 2 * i, 2).c_str(), NULL, 16);
    }
}

static bool check_hex(const char *name, const unsigned char *got, const char *want_hex, size_t n) {
    ++test_checks;
    std::string got_hex = to_hex(got, n);
    if (got_hex == want_hex) {
        return true;
    }
    ++test_failures;
    fprintf(stderr, "FAIL %s\n  got : %s\n  want: %s\n", name, got_hex.c_str(), want_hex);
    return false;
}

static bool check_int(const char *name, long long got, long long want) {
    ++test_checks;
    if (got == want) {
        return true;
    }
    ++test_failures;
    fprintf(stderr, "FAIL %s\n  got : %lld\n  want: %lld\n", name, got, want);
    return false;
}

static void fill_message(unsigned char *out, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        out[i] = (unsigned char)((i * 181 + 37) & 0xFF);
    }
}

#endif
