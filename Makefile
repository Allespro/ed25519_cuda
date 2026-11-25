CXX ?= g++
NVCC ?= nvcc
CPU_SHIM := tests/cpu_sim/cuda_cpu_sim.h

LIB_CU := fe.cu sc.cu ge.cu sha512.cu sha3.cu keccakf.cu onion.cu add_scalar.cu key_exchange.cu keypair.cu seed.cu sign.cu verify.cu

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
SANITIZE_FLAGS := -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra

.PHONY: all test test-sanitize test-gpu clean

all: test

build:
	mkdir -p build

HEADERS := tests/common.h tests/launch.h tests/test_vectors.h $(CPU_SHIM) \
	ed25519.cuh fe.cuh fixedint.cuh ge.cuh keccakf.cuh onion.cuh precomp_data.cuh sc.cuh sha3.cuh sha512.cuh
CPU_SOURCES := test.cu $(LIB_CU) tests/cpu_sim/cuda_cpu_sim.cpp

build/test_cpu: $(CPU_SOURCES) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -I. -include $(CPU_SHIM) -x c++ $(CPU_SOURCES) -o $@

build/test_sanitized: $(CPU_SOURCES) $(HEADERS) | build
	$(CXX) $(SANITIZE_FLAGS) -I. -include $(CPU_SHIM) -x c++ $(CPU_SOURCES) -o $@

build/test_gpu: test.cu $(LIB_CU) $(HEADERS) | build
	$(NVCC) -rdc=true -std=c++17 -I. test.cu $(LIB_CU) -o $@

test: build/test_cpu
	./build/test_cpu

test-sanitize: build/test_sanitized
	./build/test_sanitized

test-gpu: build/test_gpu
	./build/test_gpu

clean:
	rm -rf build

tests/test_vectors.h: tests/gen_vectors.py
	python3 tests/gen_vectors.py > $@
