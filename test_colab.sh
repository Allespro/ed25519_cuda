#!/bin/bash
echo "Compiling CUDA code for tagets - Nvidia Tesla K80 and Nvidia Tesla T4"
time nvcc *.cu -rdc=true -arch=sm_75 -gencode=arch=compute_37,code=sm_37 -Wno-deprecated-gpu-targets
echo "Run tests - with logging"
time ./a.out --logging
echo "Run test with memory profiler to check for memory error"
cuda-memcheck ./a.out --logging

