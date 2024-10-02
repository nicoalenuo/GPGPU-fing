
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>

using namespace std;

#define FILAS 1024
#define COLUMNAS 1024
#define TAM_BLOQUE_X 32
#define TAM_BLOQUE_Y 32

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void transpose_kernel(int* d_matrix, int* d_transposed_matrix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int id = COLUMNAS * x + y;
	int id_transpuesta = COLUMNAS * y + x;

	d_transposed_matrix[id_transpuesta] = d_matrix[id];
}

int main(int argc, char* argv[]){
	size_t tam_matriz = FILAS * COLUMNAS * sizeof(int);

	int* h_matrix;
	int* d_matrix;
	int* d_transposed_matrix;

	h_matrix = (int*)malloc(tam_matriz);

	for (int i = 0; i < FILAS * COLUMNAS; i++) 
		h_matrix[i] = i;

	CUDA_CHK(cudaMalloc((void**)&d_matrix, tam_matriz));
	CUDA_CHK(cudaMalloc((void**)&d_transposed_matrix, tam_matriz));

	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, tam_matriz, cudaMemcpyHostToDevice));

	dim3 tamGrid1(FILAS / TAM_BLOQUE_X, COLUMNAS / TAM_BLOQUE_Y);
	dim3 tamBlock1(TAM_BLOQUE_X, TAM_BLOQUE_Y);

	for (int i = 0; i < 10; i++) {
		transpose_kernel << <tamGrid1, tamBlock1 >> > (d_matrix, d_transposed_matrix);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}

	CUDA_CHK(cudaMemcpy(h_matrix, d_transposed_matrix, tam_matriz, cudaMemcpyDeviceToHost));

	free(h_matrix);
	CUDA_CHK(cudaFree(d_matrix));
	CUDA_CHK(cudaFree(d_transposed_matrix));
	
	return 0;
}


