#!/usr/bin/env python3
import base64
import hashlib
import random

q = 2**255 - 19
l = 2**252 + 27742317777372353535851937790883648493
M64 = (1 << 64) - 1


def H(m):
    return hashlib.sha512(m).digest()


def inv(x):
    return pow(x, q - 2, q)


d = -121665 * inv(121666) % q
I = pow(2, (q - 1) // 4, q)


def xrecover(y):
    xx = (y * y - 1) * inv(d * y * y + 1)
    x = pow(xx, (q + 3) // 8, q)
    if (x * x - xx) % q != 0:
        x = (x * I) % q
    if x % 2 != 0:
        x = q - x
    return x


By = 4 * inv(5)
Bx = xrecover(By)
B = [Bx % q, By % q]
IDENT = [0, 1]


def edwards_add(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2)
    return [x3 % q, y3 % q]


def scalarmult(P, e):
    if e == 0:
        return IDENT
    Q = scalarmult(P, e >> 1)
    Q = edwards_add(Q, Q)
    if e & 1:
        Q = edwards_add(Q, P)
    return Q


def encodepoint(P):
    x, y = P
    bs = y.to_bytes(32, 'little')
    return bs[:31] + bytes([bs[31] ^ ((x & 1) << 7)])


def secret_expand(seed):
    h = H(seed)
    a = int.from_bytes(h[:32], 'little')
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h


def publickey(seed):
    a, _ = secret_expand(seed)
    return encodepoint(scalarmult(B, a))


def signature(seed, msg):
    a, h = secret_expand(seed)
    pk = encodepoint(scalarmult(B, a))
    r = int.from_bytes(H(h[32:] + msg), 'little') % l
    R = encodepoint(scalarmult(B, r))
    k = int.from_bytes(H(R + pk + msg), 'little') % l
    s = (r + k * a) % l
    return R + s.to_bytes(32, 'little')


def clamp(k):
    k = bytearray(k[:32])
    k[0] &= 248
    k[31] &= 63
    k[31] |= 64
    return int.from_bytes(k, 'little')


def x25519(k_bytes, u_bytes):
    return x25519_raw(clamp(k_bytes), u_bytes)


def x25519_raw(k, u_bytes):
    x1 = int.from_bytes(u_bytes, 'little') & ((1 << 255) - 1)
    x2, z2, x3, z3, swap = 1, 0, x1, 1, 0
    for t in range(254, -1, -1):
        kt = (k >> t) & 1
        if swap ^ kt:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = kt
        A = (x2 + z2) % q
        AA = A * A % q
        Bb = (x2 - z2) % q
        BB = Bb * Bb % q
        E = (AA - BB) % q
        C = (x3 + z3) % q
        D = (x3 - z3) % q
        DA = D * A % q
        CB = C * Bb % q
        x3 = (DA + CB) % q
        x3 = x3 * x3 % q
        z3 = (DA - CB) % q
        z3 = z3 * z3 % q
        z3 = z3 * x1 % q
        x2 = AA * BB % q
        z2 = E * (AA + a24() * E) % q
    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2
    return (x2 * pow(z2, q - 2, q) % q).to_bytes(32, 'little')


def a24():
    return 121665


def u_to_public_key(u_bytes):
    u = int.from_bytes(u_bytes, 'little') & ((1 << 255) - 1)
    y = (u - 1) * inv(u + 1) % q
    return encodepoint([xrecover(y), y])


def add_scalar(pk, sk, scalar):
    n = int.from_bytes(bytearray(scalar[:31]) + bytes([scalar[31] & 127]), 'little')
    a = int.from_bytes(sk[:32], 'little')
    new_a = (a + n) % l
    new_pk = encodepoint(scalarmult(B, new_a))
    new_sk = new_a.to_bytes(32, 'little') + H(sk[32:64] + scalar)[:32]
    return new_pk, new_sk


def onion_address(pk):
    checksum = hashlib.sha3_256(b".onion checksum" + pk + b"\x03").digest()[:2]
    return base64.b32encode(pk + checksum + b"\x03").decode().lower() + ".onion", checksum


RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
      0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
      0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
      0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
      0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]


def rol(x, n):
    return ((x << n) | (x >> (64 - n))) & M64


def keccakf(state):
    st = list(state)
    for rnd in range(24):
        bc = [st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20] for x in range(5)]
        for x in range(5):
            t = bc[(x + 4) % 5] ^ rol(bc[(x + 1) % 5], 1)
            for y in range(0, 25, 5):
                st[y + x] ^= t
        t = st[1]
        for i in range(24):
            j = PILN[i]
            bc[0] = st[j]
            st[j] = rol(t, ROTC[i])
            t = bc[0]
        for y in range(0, 25, 5):
            for x in range(5):
                bc[x] = st[y + x]
            for x in range(5):
                st[y + x] ^= (~bc[(x + 1) % 5]) & bc[(x + 2) % 5]
        st[0] ^= RC[rnd]
    return st


def words_hex(words):
    return b''.join(w.to_bytes(8, 'little') for w in words).hex()


def hash_message(n):
    return bytes((i * 181 + 37) & 0xFF for i in range(n))


def main():
    rng = random.Random(0xED25519)
    out = []
    w = out.append

    w("#ifndef ED25519_TEST_VECTORS_H")
    w("#define ED25519_TEST_VECTORS_H")
    w("")
    w("static const struct { const char *seed, *msg, *pk, *sig; } RFC8032[] = {")
    rfc = [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60", "",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb", "72",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7", "af82",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ]
    msg1023 = hash_message(1023)
    seed1023 = bytes(rng.randrange(256) for _ in range(32))
    rfc.append((seed1023.hex(), msg1023.hex(), publickey(seed1023).hex(), signature(seed1023, msg1023).hex()))
    for seed, msg, pk, sig in rfc:
        assert publickey(bytes.fromhex(seed)).hex() == pk, "RFC 8032 public key mismatch"
        assert signature(bytes.fromhex(seed), bytes.fromhex(msg)).hex() == sig, "RFC 8032 signature mismatch"
        w('    {"%s", "%s", "%s", "%s"},' % (seed, msg, pk, sig))
    w("};")
    w("")

    w("static const struct { unsigned int msg_len; const char *seed, *pk, *sig; } SIGN[] = {")
    for n in [1, 31, 32, 33, 63, 64, 111, 112, 113, 127, 128, 129, 200, 255, 256, 257, 1000]:
        msg = hash_message(n)
        seed = bytes(rng.randrange(256) for _ in range(32))
        w('    {%d, "%s", "%s", "%s"},' % (n, seed.hex(), publickey(seed).hex(), signature(seed, msg).hex()))
    w("};")
    w("")

    lengths = list(range(0, 21)) + list(range(104, 146)) + list(range(268, 277)) + [511, 512, 1000]
    w("static const struct { unsigned int msg_len; const char *seed, *pk, *sig; } SIGN_ONE_KEY[] = {")
    one_seed = bytes(rng.randrange(256) for _ in range(32))
    for n in [0, 1, 31, 32, 33, 63, 64, 111, 112, 113, 127, 128, 129, 200, 255, 256, 257, 1000]:
        w('    {%d, "%s", "%s", "%s"},' % (n, one_seed.hex(), publickey(one_seed).hex(),
                                           signature(one_seed, hash_message(n)).hex()))
    w("};")
    w("")
    w("static const unsigned int HASH_LENGTHS[] = {%s};" % ", ".join(str(n) for n in lengths))
    w("static const char *SHA512_VECTORS[] = {")
    for n in lengths:
        w('    "%s",' % hashlib.sha512(hash_message(n)).hexdigest())
    w("};")
    w("static const char *SHA3_256_VECTORS[] = {")
    for n in lengths:
        w('    "%s",' % hashlib.sha3_256(hash_message(n)).hexdigest())
    w("};")
    w("")

    states = [[0] * 25, list(range(25)), [rng.getrandbits(64) for _ in range(25)]]
    w("static const char *KECCAKF_IN[] = {")
    for s in states:
        w('    "%s",' % words_hex(s))
    w("};")
    w("static const char *KECCAKF_OUT[] = {")
    for s in states:
        w('    "%s",' % words_hex(keccakf(s)))
    w("};")
    w("")

    p = q
    w("static const struct { const char *a, *b, *a_mod, *b_mod, *sum, *diff, *prod, *a_sq, *a_sq2, *a_neg, *a_121666, "
      "*a_pow22523, *a_inv, *b_inv, *a_odd; } FE[] = {")
    fe_inputs = [b"\x00" * 32, b"\x01" + b"\x00" * 31, (p - 1).to_bytes(32, 'little'), p.to_bytes(32, 'little'),
                 (p + 1).to_bytes(32, 'little'), b"\xff" * 32]
    fe_inputs += [rng.randbytes(32) for _ in range(18)]
    for i in range(0, len(fe_inputs), 2):
        a_bytes, b_bytes = fe_inputs[i], fe_inputs[i + 1]
        a = int.from_bytes(a_bytes, 'little') % (1 << 255) % p
        b = int.from_bytes(b_bytes, 'little') % (1 << 255) % p
        e = (p - 5) // 8
        w('    {"%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%d"},' % (
            a_bytes.hex(), b_bytes.hex(),
            a.to_bytes(32, 'little').hex(), b.to_bytes(32, 'little').hex(),
            ((a + b) % p).to_bytes(32, 'little').hex(), ((a - b) % p).to_bytes(32, 'little').hex(),
            (a * b % p).to_bytes(32, 'little').hex(), (a * a % p).to_bytes(32, 'little').hex(),
            (2 * a * a % p).to_bytes(32, 'little').hex(), ((-a) % p).to_bytes(32, 'little').hex(),
            (a * 121666 % p).to_bytes(32, 'little').hex(),
            pow(a, e, p).to_bytes(32, 'little').hex(),
            (inv(a) if a else 0).to_bytes(32, 'little').hex(), (inv(b) if b else 0).to_bytes(32, 'little').hex(),
            a & 1))
    w("};")
    w("")

    w("static const struct { const char *s, *reduced; } SC_REDUCE[] = {")
    sc_inputs = [b"\x00" * 64, b"\x01" + b"\x00" * 63, b"\xff" * 64,
                 l.to_bytes(64, 'little'), (l - 1).to_bytes(64, 'little'), (l + 1).to_bytes(64, 'little'),
                 (l * l).to_bytes(64, 'little')]
    sc_inputs += [rng.randbytes(64) for _ in range(5)]
    for s_bytes in sc_inputs:
        w('    {"%s", "%s"},' % (s_bytes.hex(), (int.from_bytes(s_bytes, 'little') % l).to_bytes(32, 'little').hex()))
    w("};")
    w("static const struct { const char *a, *b, *c, *out; } SC_MULADD[] = {")
    for _ in range(8):
        a_b, b_b, c_b = rng.randbytes(32), rng.randbytes(32), rng.randbytes(32)
        r = (int.from_bytes(a_b, 'little') * int.from_bytes(b_b, 'little') + int.from_bytes(c_b, 'little')) % l
        w('    {"%s", "%s", "%s", "%s"},' % (a_b.hex(), b_b.hex(), c_b.hex(), r.to_bytes(32, 'little').hex()))
    w("};")
    w("")

    w("static const struct { const char *scalar, *point; } SCALARMULT[] = {")
    scalars = [0, 1, 2, l, l - 1, l + 1, 2 * l, (1 << 252), (1 << 253) - 1, (1 << 254) - 8, ((1 << 254) - 8) | 1,
               (1 << 127) - 1, (1 << 254) | 1, (0x7FFFFFFF << 224) | 1]
    scalars += [int.from_bytes(rng.randbytes(32), 'little') & 0x7FFFFFFFFFFFFFFF for _ in range(6)]
    for s in scalars:
        assert s < 1 << 255
        w('    {"%s", "%s"},' % (s.to_bytes(32, 'little').hex(), encodepoint(scalarmult(B, s)).hex()))
    w("};")
    w("")

    for label, enc in (("SCALARMULT_LARGE_Y", bytes([0xAA] * 31 + [0x7F])), ("SCALARMULT_MAX_Y", bytes([0xFF] * 31 + [0x7F]))):
        yy = int.from_bytes(enc, 'little') % (1 << 255) % q
        w('static const char *%s = "%s";' % (label, encodepoint([xrecover(yy), yy]).hex()))
    w("")

    def emit_add_scalar(name, scalar_of):
        w("static const struct { const char *pk, *sk, *scalar, *pk_out, *sk_out; } %s[] = {" % name)
        for i in range(4):
            seed = rng.randbytes(32)
            a, h = secret_expand(seed)
            pk = publickey(seed)
            sk = a.to_bytes(32, 'little') + h[32:64]
            scalar = scalar_of(i)
            new_pk, new_sk = add_scalar(pk, sk, scalar)
            assert publickey_of(new_sk) == new_pk, "add_scalar key pair is inconsistent"
            w('    {"%s", "%s", "%s", "%s", "%s"},' % (pk.hex(), sk.hex(), scalar.hex(), new_pk.hex(), new_sk.hex()))
        w("};")
        w("")

    shared_scalar = rng.randbytes(32)
    emit_add_scalar("ADD_SCALAR", lambda i: shared_scalar)
    emit_add_scalar("ADD_MULTI_SCALAR", lambda i: rng.randbytes(32))

    w("static const struct { const char *priv, *pk, *secret; } KEY_EXCHANGE[] = {")
    ka = bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
    kb = bytes.fromhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
    ua = bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
    ub = bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
    shared = "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"
    assert x25519(ka, ub).hex() == shared and x25519(kb, ua).hex() == shared, "RFC 7748 vector mismatch"
    assert x25519(ka, (9).to_bytes(32, 'little')).hex() == ua.hex(), "RFC 7748 public key mismatch"
    w('    {"%s", "%s", "%s"},' % (ka.hex(), u_to_public_key(ub).hex(), x25519(ka, ub).hex()))
    w('    {"%s", "%s", "%s"},' % (kb.hex(), u_to_public_key(ua).hex(), x25519(kb, ua).hex()))
    for _ in range(4):
        seed = rng.randbytes(32)
        a, _ = secret_expand(seed)
        k_a = a.to_bytes(32, 'little')
        u_b = rng.randbytes(32)
        w('    {"%s", "%s", "%s"},' % (k_a.hex(), u_to_public_key(u_b).hex(), x25519(k_a, u_b).hex()))
    w("};")
    w("")

    w("static const struct { const char *pk, *checksum, *address; } ONION_ADDRESSES[] = {")
    onion_keys = [bytes.fromhex(pk) for _, _, pk, _ in rfc]
    onion_keys += [publickey(rng.randbytes(32)) for _ in range(4)]
    for pk in onion_keys:
        addr, checksum = onion_address(pk)
        w('    {"%s", "%s", "%s"},' % (pk.hex(), checksum.hex(), addr))
    w("};")
    w("")
    w("#endif")

    sys.stdout.write("\n".join(out) + "\n")


def publickey_of(sk):
    return encodepoint(scalarmult(B, int.from_bytes(sk[:32], 'little')))


if __name__ == "__main__":
    import sys
    main()
