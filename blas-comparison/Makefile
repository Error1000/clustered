openblas:
	gcc -O2 -lopenblas_64 -lpthread sgemm_openblas.c

cublas:
	g++ -O2 -I/opt/cuda/targets/x86_64-linux/include -L/opt/cuda/targets/x86_64-linux/lib -lcublas -lcudart gemm_cublas.cpp
