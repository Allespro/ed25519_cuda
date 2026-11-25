#!/bin/bash
ARCH="${1:-sm_75}"

time nvcc *.cu -rdc=true -arch="$ARCH" -Wno-deprecated-gpu-targets -o test_gpu
time ./test_gpu
