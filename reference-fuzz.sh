#!/bin/bash
python fuzzer.py --n_iter 1000 target/release/matrix-multiply-cpu-reference $1
