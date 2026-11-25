#include <algorithm>
#include "tests/common.h"
#include <vector>

static void run_keccakf_tests() {
    unsigned char input[200];
    unsigned char expected[200];
    unsigned char got[200];
    uint64_t state[25];
    for (size_t i = 0; i < sizeof(KECCAKF_IN) / sizeof(KECCAKF_IN[0]); ++i) {
        unhex(KECCAKF_IN[i], input, 200);
        unhex(KECCAKF_OUT[i], expected, 200);
        for (int j = 0; j < 25; ++j) {
            state[j] = 0;
            for (int k = 0; k < 8; ++k) {
                state[j] |= (uint64_t)input[j * 8 + k] << (8 * k);
            }
        }
        keccakf(state);
        for (int j = 0; j < 25; ++j) {
            for (int k = 0; k < 8; ++k) {
                got[j * 8 + k] = (unsigned char)(state[j] >> (8 * k));
            }
        }
        check_hex("keccakf", got, KECCAKF_OUT[i], 200);
        (void)expected;
    }
}

static void run_sha512_tests() {
    unsigned char message[1000];
    unsigned char got[64];
    const size_t count = sizeof(HASH_LENGTHS) / sizeof(HASH_LENGTHS[0]);
    for (size_t i = 0; i < count; ++i) {
        fill_message(message, HASH_LENGTHS[i]);
        sha512(message, HASH_LENGTHS[i], got);
        check_hex("sha512", got, SHA512_VECTORS[i], 64);

        sha512_context context;
        unsigned int fed = 0;
        unsigned int chunk = 1 + (unsigned int)(i % 13);
        sha512_init(&context);
        while (fed < HASH_LENGTHS[i]) {
            unsigned int n = HASH_LENGTHS[i] - fed;
            if (n > chunk) {
                n = chunk;
            }
            sha512_update(&context, message + fed, n);
            fed += n;
        }
        sha512_final(&context, got);
        check_hex("sha512 streamed", got, SHA512_VECTORS[i], 64);
    }
}

struct MessagePack {
    std::vector<unsigned char> data;
    std::vector<size_t> offsets;
};

static void finish_pack(MessagePack &pack) {
    pack.offsets.push_back(pack.data.size());
}

static void run_keypair_tests() {
    const size_t count = sizeof(RFC8032) / sizeof(RFC8032[0]);
    std::vector<unsigned char> seed(32 * count), public_key(32 * count), private_key(64 * count);
    for (size_t i = 0; i < count; ++i) {
        unhex(RFC8032[i].seed, &seed[i * 32], 32);
    }
    launch(ed25519_kernel_create_keypair_batch, 2, 3, &public_key[0], &private_key[0], &seed[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_hex("RFC8032 public key", &public_key[i * 32], RFC8032[i].pk, 32);
    }
    for (size_t i = 0; i < count; ++i) {
        seed[i * 32] ^= 0xFF;
    }
    launch(ed25519_kernel_create_keypair_batch, 1, 1, &public_key[0], &private_key[0], &seed[0], 0);
    for (size_t i = 0; i < count; ++i) {
        check_hex("limit=0 leaves output untouched", &public_key[i * 32], RFC8032[i].pk, 32);
    }
}

static void run_single_keypair_sign_tests() {
    const size_t count = sizeof(SIGN_ONE_KEY) / sizeof(SIGN_ONE_KEY[0]);
    unsigned char seed[32], public_key[32], private_key[64];
    MessagePack pack;
    std::vector<unsigned char> signature(64 * count), message(1000);
    unhex(SIGN_ONE_KEY[0].seed, seed, 32);
    launch(ed25519_kernel_create_keypair_batch, 1, 1, public_key, private_key, seed, 1);
    check_hex("SIGN_ONE_KEY public key", public_key, SIGN_ONE_KEY[0].pk, 32);
    for (size_t i = 0; i < count; ++i) {
        fill_message(&message[0], SIGN_ONE_KEY[i].msg_len);
        pack.offsets.push_back(pack.data.size());
        pack.data.insert(pack.data.end(), message.begin(), message.begin() + SIGN_ONE_KEY[i].msg_len);
    }
    finish_pack(pack);
    launch(ed25519_kernel_sign_batch_single_keypair, 3, 8, &signature[0], pack.data.empty() ? NULL : &pack.data[0], &pack.offsets[0], public_key, private_key, (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_hex("SIGN_ONE_KEY signature", &signature[i * 64], SIGN_ONE_KEY[i].sig, 64);
    }
}

static void run_multi_keypair_sign_tests() {
    const size_t count = sizeof(SIGN) / sizeof(SIGN[0]);
    std::vector<unsigned char> seed(32 * count), public_key(32 * count), private_key(64 * count), signature(64 * count);
    std::vector<int> mapping(count);
    MessagePack pack;
    for (size_t i = 0; i < count; ++i) {
        unhex(SIGN[i].seed, &seed[i * 32], 32);
        std::vector<unsigned char> message(SIGN[i].msg_len);
        fill_message(&message[0], SIGN[i].msg_len);
        pack.offsets.push_back(pack.data.size());
        pack.data.insert(pack.data.end(), message.begin(), message.end());
        mapping[i] = (int)i;
    }
    finish_pack(pack);
    launch(ed25519_kernel_create_keypair_batch, 1, (unsigned int)count, &public_key[0], &private_key[0], &seed[0], (int)count);
    launch(ed25519_kernel_sign_batch_multi_keypair, 2, 16, &signature[0], &pack.data[0], &pack.offsets[0], &public_key[0], &private_key[0], &mapping[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_hex("SIGN public key", &public_key[i * 32], SIGN[i].pk, 32);
        check_hex("SIGN signature", &signature[i * 64], SIGN[i].sig, 64);
    }
}

static void run_verify_tests() {
    const size_t count = sizeof(SIGN_ONE_KEY) / sizeof(SIGN_ONE_KEY[0]);
    unsigned char seed[32], public_key[32], private_key[64];
    MessagePack pack;
    std::vector<unsigned char> signature(64 * count), message(1000);
    std::vector<int> verified(count);
    unhex(SIGN_ONE_KEY[0].seed, seed, 32);
    for (size_t i = 0; i < count; ++i) {
        fill_message(&message[0], SIGN_ONE_KEY[i].msg_len);
        pack.offsets.push_back(pack.data.size());
        pack.data.insert(pack.data.end(), message.begin(), message.begin() + SIGN_ONE_KEY[i].msg_len);
        unhex(SIGN_ONE_KEY[i].sig, &signature[i * 64], 64);
    }
    finish_pack(pack);
    launch(ed25519_kernel_create_keypair_batch, 1, 1, public_key, private_key, seed, 1);

    launch(ed25519_kernel_verify_batch_single_keypair, 4, 8, &signature[0], &pack.data[0], &pack.offsets[0], public_key, &verified[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_int("valid signature verifies", verified[i], 1);
    }

    signature[0 * 64 + 10] ^= 0x01;
    launch(ed25519_kernel_verify_batch_single_keypair, 1, (unsigned int)count, &signature[0], &pack.data[0], &pack.offsets[0], public_key, &verified[0], (int)count);
    check_int("flipped signature bit rejected", verified[0], 0);
    for (size_t i = 1; i < count; ++i) {
        check_int("other signatures still verify", verified[i], 1);
    }
    unhex(SIGN_ONE_KEY[0].sig, &signature[0], 64);

    std::vector<unsigned char> wrong_key(32);
    unhex(SIGN[0].pk, &wrong_key[0], 32);
    launch(ed25519_kernel_verify_batch_single_keypair, 1, (unsigned int)count, &signature[0], &pack.data[0], &pack.offsets[0], &wrong_key[0], &verified[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_int("wrong public key rejected", verified[i], 0);
    }

    std::vector<unsigned char> tampered(pack.data);
    for (size_t i = 0; i < tampered.size(); ++i) {
        tampered[i] ^= 0x5A;
    }
    launch(ed25519_kernel_verify_batch_single_keypair, 1, (unsigned int)count, &signature[0], &tampered[0], &pack.offsets[0], public_key, &verified[0], (int)count);
    for (size_t i = 1; i < count; ++i) {
        check_int("tampered message rejected", verified[i], 0);
    }

    std::vector<unsigned char> high_s(64 * count);
    memcpy(&high_s[0], &signature[0], 64 * count);
    high_s[63] |= 0xE0;
    launch(ed25519_kernel_verify_batch_single_keypair, 1, 1, &high_s[0], &pack.data[0], &pack.offsets[0], public_key, &verified[0], 1);
    check_int("non-canonical S rejected", verified[0], 0);

    std::vector<unsigned char> junk_key(32, 0xAA);
    junk_key[31] = 0x7F;
    launch(ed25519_kernel_verify_batch_single_keypair, 1, 1, &signature[0], &pack.data[0], &pack.offsets[0], &junk_key[0], &verified[0], 1);
    check_int("unparseable public key rejected", verified[0], 0);
}

static void run_fe_tests() {
    const size_t count = sizeof(FE) / sizeof(FE[0]);
    unsigned char bytes_a[32], bytes_b[32];
    for (size_t i = 0; i < count; ++i) {
        unhex(FE[i].a, bytes_a, 32);
        unhex(FE[i].b, bytes_b, 32);
        fe a, b, r, s;
        fe_frombytes(a, bytes_a);
        fe_frombytes(b, bytes_b);
        fe_tobytes(bytes_a, a);
        check_hex("fe roundtrip a", bytes_a, FE[i].a_mod, 32);
        fe_tobytes(bytes_b, b);
        check_hex("fe roundtrip b", bytes_b, FE[i].b_mod, 32);

        fe_add(r, a, b);
        fe_tobytes(bytes_a, r);
        check_hex("fe_add", bytes_a, FE[i].sum, 32);

        fe_sub(r, a, b);
        fe_tobytes(bytes_a, r);
        check_hex("fe_sub", bytes_a, FE[i].diff, 32);

        fe_mul(r, a, b);
        fe_tobytes(bytes_a, r);
        check_hex("fe_mul", bytes_a, FE[i].prod, 32);

        fe_sq(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_sq", bytes_a, FE[i].a_sq, 32);

        fe_sq2(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_sq2", bytes_a, FE[i].a_sq2, 32);

        fe_neg(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_neg", bytes_a, FE[i].a_neg, 32);

        fe_mul121666(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_mul121666", bytes_a, FE[i].a_121666, 32);

        fe_pow22523(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_pow22523", bytes_a, FE[i].a_pow22523, 32);

        fe_invert(r, a);
        fe_tobytes(bytes_a, r);
        check_hex("fe_invert a", bytes_a, FE[i].a_inv, 32);

        fe_invert(r, b);
        fe_tobytes(bytes_a, r);
        check_hex("fe_invert b", bytes_a, FE[i].b_inv, 32);

        check_int("fe_isnegative", fe_isnegative(a), (long long)strtoul(std::string(FE[i].a_odd).c_str(), NULL, 10));
        check_int("fe_isnonzero", fe_isnonzero(a), strcmp(FE[i].a_mod, "0000000000000000000000000000000000000000000000000000000000000000") != 0);

        unsigned char flipped[32];
        unhex(FE[i].b, flipped, 32);
        flipped[31] ^= 0x80;
        fe_frombytes(s, flipped);
        fe_tobytes(flipped, s);
        check_hex("fe_frombytes ignores top bit", flipped, FE[i].b_mod, 32);

        fe_copy(r, b);
        fe_cmov(r, a, 0);
        fe_tobytes(flipped, r);
        check_hex("fe_cmov b=0 keeps", flipped, FE[i].b_mod, 32);
        fe_cmov(r, a, 1);
        fe_tobytes(flipped, r);
        check_hex("fe_cmov b=1 takes", flipped, FE[i].a_mod, 32);

        fe_copy(r, a);
        fe_copy(s, b);
        fe_cswap(r, s, 1);
        fe_tobytes(flipped, r);
        check_hex("fe_cswap b=1 swaps", flipped, FE[i].b_mod, 32);
        fe_tobytes(bytes_a, s);
        check_hex("fe_cswap b=1 swaps", bytes_a, FE[i].a_mod, 32);
        fe_cswap(r, s, 0);
        fe_tobytes(flipped, r);
        check_hex("fe_cswap b=0 keeps", flipped, FE[i].b_mod, 32);

        fe_1(r);
        fe_tobytes(flipped, r);
        check_hex("fe_1", flipped, "0100000000000000000000000000000000000000000000000000000000000000", 32);
    }
}

static void run_sc_tests() {
    const size_t reduce_count = sizeof(SC_REDUCE) / sizeof(SC_REDUCE[0]);
    unsigned char s[64], got[32];
    for (size_t i = 0; i < reduce_count; ++i) {
        unhex(SC_REDUCE[i].s, s, 64);
        sc_reduce(s);
        check_hex("sc_reduce", s, SC_REDUCE[i].reduced, 32);
    }
    const size_t muladd_count = sizeof(SC_MULADD) / sizeof(SC_MULADD[0]);
    unsigned char a[32], b[32], c[32];
    for (size_t i = 0; i < muladd_count; ++i) {
        unhex(SC_MULADD[i].a, a, 32);
        unhex(SC_MULADD[i].b, b, 32);
        unhex(SC_MULADD[i].c, c, 32);
        sc_muladd(got, a, b, c);
        check_hex("sc_muladd", got, SC_MULADD[i].out, 32);
    }
}

static void ge_negate_undo(ge_p3 *point) {
    fe_neg(point->X, point->X);
    fe_neg(point->T, point->T);
}

static void run_ge_tests() {
    const size_t count = sizeof(SCALARMULT) / sizeof(SCALARMULT[0]);
    unsigned char scalar[32], got[32];
    ge_p3 point;
    for (size_t i = 0; i < count; ++i) {
        unhex(SCALARMULT[i].scalar, scalar, 32);
        ge_scalarmult_base(&point, scalar);
        ge_p3_tobytes(got, &point);
        check_hex("ge_scalarmult_base", got, SCALARMULT[i].point, 32);

        ge_p3 parsed;
        int rc = ge_frombytes_negate_vartime(&parsed, got);
        check_int("valid point parses", rc, 0);
        fe_neg(parsed.X, parsed.X);
        fe_neg(parsed.T, parsed.T);
        ge_p3_tobytes(got, &parsed);
        check_hex("point roundtrip", got, SCALARMULT[i].point, 32);
    }
    unsigned char encoding[32];
    int invalid_y[] = {2, 7, 8, 11};
    for (size_t i = 0; i < sizeof(invalid_y) / sizeof(invalid_y[0]); ++i) {
        memset(encoding, 0, 32);
        encoding[0] = (unsigned char)invalid_y[i];
        check_int("non-residue y rejected", ge_frombytes_negate_vartime(&point, encoding), -1);
    }
    memset(encoding, 0xAA, 31);
    encoding[31] = 0x7F;
    check_int("large valid y accepted", ge_frombytes_negate_vartime(&point, encoding), 0);
    ge_negate_undo(&point);
    ge_p3_tobytes(encoding, &point);
    check_hex("large y roundtrip", encoding, SCALARMULT_LARGE_Y, 32);
    memset(encoding, 0xFF, 31);
    encoding[31] = 0x7F;
    check_int("max y accepted", ge_frombytes_negate_vartime(&point, encoding), 0);
    ge_negate_undo(&point);
    ge_p3_tobytes(encoding, &point);
    check_hex("max y roundtrip", encoding, SCALARMULT_MAX_Y, 32);
}


static void run_key_exchange_tests() {
    const size_t count = sizeof(KEY_EXCHANGE) / sizeof(KEY_EXCHANGE[0]);
    std::vector<unsigned char> private_key(64 * count, 0), public_key(32 * count), secret(32 * count);
    for (size_t i = 0; i < count; ++i) {
        unhex(KEY_EXCHANGE[i].priv, &private_key[i * 64], 32);
        unhex(KEY_EXCHANGE[i].pk, &public_key[i * 32], 32);
    }
    launch(ed25519_kernel_key_exchange_batch, 1, (unsigned int)count, &secret[0], &public_key[0], &private_key[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_hex("key_exchange secret", &secret[i * 32], KEY_EXCHANGE[i].secret, 32);
    }
    unsigned char seed[64], seeded_private[128], own_public[64], own_secret[32], peer_secret[32];
    unhex(RFC8032[0].seed, seed, 32);
    unhex(RFC8032[1].seed, seed + 32, 32);
    launch(ed25519_kernel_create_keypair_batch, 1, 2, own_public, seeded_private, seed, 2);
    launch(ed25519_kernel_key_exchange_batch, 1, 1, own_secret, own_public + 32, seeded_private, 1);
    launch(ed25519_kernel_key_exchange_batch, 1, 1, peer_secret, own_public, seeded_private + 64, 1);
    check_hex("key_exchange is symmetric", own_secret, to_hex(peer_secret, 32).c_str(), 32);
}

static void run_add_scalar_tests() {
    const size_t count = sizeof(ADD_SCALAR) / sizeof(ADD_SCALAR[0]);
    std::vector<unsigned char> public_key(32 * count), private_key(64 * count), scalar(32);
    for (size_t i = 0; i < count; ++i) {
        unhex(ADD_SCALAR[i].pk, &public_key[i * 32], 32);
        unhex(ADD_SCALAR[i].sk, &private_key[i * 64], 64);
    }
    unhex(ADD_SCALAR[0].scalar, &scalar[0], 32);
    launch(ed25519_kernel_add_scalar_batch, 1, (unsigned int)count, &public_key[0], &private_key[0], &scalar[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "add_scalar[%d] public key", (int)i);
        check_hex(label, &public_key[i * 32], ADD_SCALAR[i].pk_out, 32);
        snprintf(label, sizeof(label), "add_scalar[%d] private key", (int)i);
        check_hex(label, &private_key[i * 64], ADD_SCALAR[i].sk_out, 64);
    }

    for (size_t i = 0; i < count; ++i) {
        unhex(ADD_SCALAR[i].pk, &public_key[i * 32], 32);
    }
    launch(ed25519_kernel_add_scalar_batch, 2, 3, &public_key[0], (unsigned char *)NULL, &scalar[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "add_scalar[%d] public-only", (int)i);
        check_hex(label, &public_key[i * 32], ADD_SCALAR[i].pk_out, 32);
    }

    const size_t multi_count = sizeof(ADD_MULTI_SCALAR) / sizeof(ADD_MULTI_SCALAR[0]);
    std::vector<unsigned char> multi_public(32 * multi_count), multi_private(64 * multi_count), multi_scalar(32 * multi_count);
    std::vector<int> mapping(multi_count);
    for (size_t i = 0; i < multi_count; ++i) {
        unhex(ADD_MULTI_SCALAR[i].pk, &multi_public[i * 32], 32);
        unhex(ADD_MULTI_SCALAR[i].sk, &multi_private[i * 64], 64);
        unhex(ADD_MULTI_SCALAR[i].scalar, &multi_scalar[i * 32], 32);
    }
    launch(ed25519_kernel_add_multi_scalar_batch, 2, 3, &multi_public[0], &multi_private[0], &multi_scalar[0], (int *)NULL, (int)multi_count);
    for (size_t i = 0; i < multi_count; ++i) {
        check_hex("add_multi_scalar NULL mapping public key", &multi_public[i * 32], ADD_MULTI_SCALAR[i].pk_out, 32);
        check_hex("add_multi_scalar NULL mapping private key", &multi_private[i * 64], ADD_MULTI_SCALAR[i].sk_out, 64);
    }
    for (size_t i = 0; i < multi_count; ++i) {
        unhex(ADD_MULTI_SCALAR[i].pk, &multi_public[i * 32], 32);
    }
    launch(ed25519_kernel_add_multi_scalar_batch, 1, (unsigned int)multi_count, &multi_public[0], (unsigned char *)NULL, &multi_scalar[0], (int *)NULL, (int)multi_count);
    for (size_t i = 0; i < multi_count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "add_multi_scalar[%d] public-only", (int)i);
        check_hex(label, &multi_public[i * 32], ADD_MULTI_SCALAR[i].pk_out, 32);
    }

    for (size_t i = 0; i < multi_count; ++i) {
        mapping[i] = (int)(multi_count - 1 - i);
        unhex(ADD_MULTI_SCALAR[mapping[i]].pk, &multi_public[i * 32], 32);
        unhex(ADD_MULTI_SCALAR[mapping[i]].sk, &multi_private[i * 64], 64);
    }
    launch(ed25519_kernel_add_multi_scalar_batch, 1, (unsigned int)multi_count, &multi_public[0], &multi_private[0], &multi_scalar[0], &mapping[0], (int)multi_count);
    for (size_t i = 0; i < multi_count; ++i) {
        check_hex("add_multi_scalar mapped public key", &multi_public[i * 32], ADD_MULTI_SCALAR[mapping[i]].pk_out, 32);
        check_hex("add_multi_scalar mapped private key", &multi_private[i * 64], ADD_MULTI_SCALAR[mapping[i]].sk_out, 64);
    }
}

static void run_sha3_256_tests() {
    unsigned char message[1000];
    unsigned char got[32];
    const size_t count = sizeof(HASH_LENGTHS) / sizeof(HASH_LENGTHS[0]);
    for (size_t i = 0; i < count; ++i) {
        fill_message(message, HASH_LENGTHS[i]);
        sha3_256(message, HASH_LENGTHS[i], got);
        check_hex("sha3_256", got, SHA3_256_VECTORS[i], 32);

        sha3_256_context context;
        unsigned int fed = 0;
        unsigned int chunk = 1 + (unsigned int)(i % 17);
        sha3_256_init(&context);
        while (fed < HASH_LENGTHS[i]) {
            unsigned int n = HASH_LENGTHS[i] - fed;
            if (n > chunk) {
                n = chunk;
            }
            sha3_256_update(&context, message + fed, n);
            fed += n;
        }
        sha3_256_final(&context, got);
        check_hex("sha3_256 streamed", got, SHA3_256_VECTORS[i], 32);
    }
}

static void run_onion_tests() {
    const size_t count = sizeof(ONION_ADDRESSES) / sizeof(ONION_ADDRESSES[0]);
    std::vector<unsigned char> public_key(32 * count);
    std::vector<char> address((ONION_ADDRESS_LEN + 1) * count, 'x');
    for (size_t i = 0; i < count; ++i) {
        unhex(ONION_ADDRESSES[i].pk, &public_key[i * 32], 32);
    }
    launch(onion_address, 3, 4, &public_key[0], &address[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        std::string got(&address[i * (ONION_ADDRESS_LEN + 1)]);
        ++test_checks;
        if (got == std::string(ONION_ADDRESSES[i].address).substr(0, ONION_ADDRESS_LEN) &&
            address[i * (ONION_ADDRESS_LEN + 1) + ONION_ADDRESS_LEN] == '\0') {
            continue;
        }
        ++test_failures;
        fprintf(stderr, "FAIL onion address[%d]\n  got : %s\n  want: %s\n", (int)i, got.c_str(),
                std::string(ONION_ADDRESSES[i].address).substr(0, ONION_ADDRESS_LEN).c_str());
    }

    const size_t seeds = sizeof(RFC8032) / sizeof(RFC8032[0]);
    std::vector<unsigned char> seed(32 * seeds), end_to_end(32 * seeds), private_key(64 * seeds);
    for (size_t i = 0; i < seeds; ++i) {
        unhex(RFC8032[i].seed, &seed[i * 32], 32);
    }
    launch(ed25519_kernel_create_keypair_batch, 2, 3, &end_to_end[0], &private_key[0], &seed[0], (int)seeds);
    launch(onion_address, 1, (unsigned int)seeds, &end_to_end[0], &address[0], (int)seeds);
    for (size_t i = 0; i < seeds; ++i) {
        std::string got(&address[i * (ONION_ADDRESS_LEN + 1)]);
        ++test_checks;
        if (got == std::string(ONION_ADDRESSES[i].address).substr(0, ONION_ADDRESS_LEN)) {
            continue;
        }
        ++test_failures;
        fprintf(stderr, "FAIL onion address end-to-end[%d]\n  got : %s\n  want: %s\n", (int)i, got.c_str(),
                std::string(ONION_ADDRESSES[i].address).substr(0, ONION_ADDRESS_LEN).c_str());
    }
}

static void run_onion_search_tests() {
    const size_t count = sizeof(ONION_ADDRESSES) / sizeof(ONION_ADDRESSES[0]);
    std::vector<unsigned char> public_key(32 * count);
    std::vector<int> found(count, -1);
    for (size_t i = 0; i < count; ++i) {
        unhex(ONION_ADDRESSES[i].pk, &public_key[i * 32], 32);
    }
    std::string first = std::string(ONION_ADDRESSES[0].address).substr(0, ONION_ADDRESS_LEN);
    char prefix[16], suffix[16], wrong[16];
    snprintf(prefix, sizeof(prefix), "%.8s", first.c_str());
    snprintf(suffix, sizeof(suffix), "%.8s", first.c_str() + ONION_ADDRESS_LEN - 8);
    snprintf(wrong, sizeof(wrong), "%.8s", std::string(ONION_ADDRESSES[1].address).substr(0, ONION_ADDRESS_LEN).c_str());

    launch(onion_search, 2, 5, &public_key[0], prefix, (char *)NULL, &found[0], (int)count);
    check_int("prefix matches its own key", found[0], 1);
    for (size_t i = 1; i < count; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "prefix of key 0 does not match key %d", (int)i);
        check_int(label, found[i], 0);
    }

    launch(onion_search, 1, (unsigned int)count, &public_key[0], (char *)NULL, suffix, &found[0], (int)count);
    check_int("suffix matches its own key", found[0], 1);
    for (size_t i = 1; i < count; ++i) {
        check_int("suffix of key 0 does not match other keys", found[i], 0);
    }

    launch(onion_search, 1, (unsigned int)count, &public_key[0], prefix, suffix, &found[0], (int)count);
    check_int("prefix+suffix match their own key", found[0], 1);
    for (size_t i = 1; i < count; ++i) {
        check_int("prefix+suffix of key 0 do not match other keys", found[i], 0);
    }

    launch(onion_search, 1, (unsigned int)count, &public_key[0], wrong, (char *)NULL, &found[0], (int)count);
    check_int("wrong prefix does not match key 0", found[0], 0);
    check_int("wrong prefix matches key 1", found[1], 1);

    launch(onion_search, 1, (unsigned int)count, &public_key[0], (char *)NULL, (char *)NULL, &found[0], (int)count);
    for (size_t i = 0; i < count; ++i) {
        check_int("empty pattern matches everything", found[i], 1);
    }

    std::fill(found.begin(), found.end(), -1);
    launch(onion_search, 1, (unsigned int)count + 3, &public_key[0], prefix, (char *)NULL, &found[0], 0);
    for (size_t i = 0; i < count; ++i) {
        check_int("limit=0 leaves output untouched", found[i], -1);
    }
}

int main() {
    run_keccakf_tests();
    run_sha3_256_tests();
    run_sha512_tests();
    run_keypair_tests();
    run_single_keypair_sign_tests();
    run_multi_keypair_sign_tests();
    run_verify_tests();
    run_fe_tests();
    run_sc_tests();
    run_ge_tests();
    run_key_exchange_tests();
    run_add_scalar_tests();
    run_onion_tests();
    run_onion_search_tests();
    printf("%d checks, %d failures\n", test_checks, test_failures);
    return test_failures ? 1 : 0;
}
