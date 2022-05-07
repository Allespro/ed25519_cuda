#!/bin/bash
time nvcc *.cu -rdc=true -arch=sm_75 -gencode=arch=compute_37,code=sm_37 -Wno-deprecated-gpu-targets
time ./a.out --logging

