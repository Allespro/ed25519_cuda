Ed25519
-------

This is a portable implementation of [Ed25519](http://ed25519.cr.yp.to/) based
on the SUPERCOP "ref10" implementation. All code is in the public domain.

All code is pure ANSI C without any dependencies, except for the random seed
generation which uses standard OS cryptography APIs. If you wish to be entirely
portable define `ED25519_NO_SEED`. This disables the `ed25519_create_seed`
function, so if your application requires key generation you must supply your
own seeding function (simply a 32 byte random number generator).


Performance
-----------

On a machine with an Intel Pentium B970 @ 2.3GHz I got the following speeds (running
on only one a single core):

    Seed + key generation:             345us
    Message signing (short message):   256us
    Message verifying (short message): 777us

The speeds on other machines may vary. Sign/verify times will be higher with
longer messages.


Usage
-----

Simply add all .c and .h files in the `src/` folder to your project and include
`ed25519.h` in any file you want to use the API. If you prefer to use a shared
library, only copy `ed25519.h` and define `ED25519_DLL` before importing. A
windows DLL is pre-built.

There are no defined types for seeds, private keys, public keys or signatures.
Instead simple `unsigned char` buffers are used with the following sizes:

```c
unsigned char seed[32];
unsigned char signature[64];
unsigned char public_key[32];
unsigned char private_key[64];
unsigned char scalar[32];
```

API
---

```c
int ed25519_create_seed(unsigned char *seed);
```

Creates a 32 byte random seed in `seed` for key generation. `seed` must be a
writable 32 byte buffer. Returns 0 on success, and nonzero on failure.

```c
void ed25519_create_keypair(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed);
```

Creates a new key pair from the given seed. `public_key` must be a writable 32
byte buffer, `private_key` must be a writable 64 byte buffer and `seed` must be
a 32 byte buffer.

```c
void ed25519_sign(unsigned char *signature,
                 const unsigned char *message, size_t message_len,
                 const unsigned char *public_key, const unsigned char *private_key);
```

Creates a signature of the given message with the given key pair. `signature`
must be a writable 64 byte buffer. `message` must have at least `message_len`
bytes to be read. 

```c
int ed25519_verify(const unsigned char *signature,
                   const unsigned char *message, size_t message_len,
                   const unsigned char *public_key);
```

Verifies the signature on the given message using `public_key`. `signature`
must be a readable 64 byte buffer. `message` must have at least `message_len`
bytes to be read. Returns 1 if the signature matches, 0 otherwise.

```c
void ed25519_add_scalar(unsigned char *public_key, unsigned char *private_key,
                        const unsigned char *scalar);
```

Adds `scalar` to the given key pair where scalar is a 32 byte buffer (possibly
generated with `ed25519_create_seed`), generating a new key pair. You can
calculate the public key sum without knowing the private key and vice versa by
passing in `NULL` for the key you don't know. This is useful for enforcing
randomness on a key pair while only knowing the public key, among other things.
Warning: the last bit of the scalar is ignored - if comparing scalars make sure
to clear it with `scalar[31] &= 127`.

Example
-------
```c
unsigned char seed[32], public_key[32], private_key[64], signature[64];
const unsigned char message[] = "TEST MESSAGE";

/* create a random seed, and a key pair out of that seed */
if (ed25519_create_seed(seed)) {
    printf("error while generating seed\n");
    exit(1);
}

ed25519_create_keypair(public_key, private_key, seed);

/* create signature on the message with the key pair */
ed25519_sign(signature, message, strlen(message), public_key, private_key);

/* verify the signature */
if (ed25519_verify(signature, message, strlen(message), public_key)) {
    printf("valid signature\n");
} else {
    printf("invalid signature\n");
}
```
