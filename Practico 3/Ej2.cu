
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>

using namespace std;

#define FILAS 4801
#define COLUMNAS 4801
#define TAM_BLOQUE_X 32
#define TAM_BLOQUE_Y 32

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void sum_kernel(int* d_matrix, int* d_sum_matrix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x + 4 < FILAS && y < COLUMNAS) {

		int id = (COLUMNAS * x) + y;
		int id_sum = (COLUMNAS * (x + 4)) + y;

		d_sum_matrix[id] = d_matrix[id] + d_matrix[id_sum];
	}
}

__global__ void sum_kernel_optimizado(int* d_matrix, int* d_sum_matrix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x + 1 < FILAS / 4 && y < COLUMNAS) { //Como en la matriz reinterpretada hay "4 veces menos elementos", sumamos solo 1 en la fila
		int id = COLUMNAS * x + y;
		int id_sum = COLUMNAS * (x + 1) + y;

		int4 elementos     = reinterpret_cast<int4*>(d_matrix)[id];
		int4 elementos_sum = reinterpret_cast<int4*>(d_matrix)[id_sum];
		int4 resultado = { 
			elementos.x + elementos_sum.x,
			elementos.y + elementos_sum.y,
			elementos.z + elementos_sum.z,
			elementos.w + elementos_sum.w 
		};
		reinterpret_cast<int4*>(d_sum_matrix)[id] = resultado;
	}
}

int main(int argc, char* argv[]) {
	size_t tam_matriz = FILAS * COLUMNAS * sizeof(int);

	int* h_matrix;
	int* d_matrix;
	int* d_sum_matrix;

	h_matrix = (int*)malloc(tam_matriz);

	for (int i = 0; i < FILAS; i++) 
		for (int j = 0; j < COLUMNAS; j++) 
			h_matrix[i * COLUMNAS + j] = i;

	CUDA_CHK(cudaMalloc((void**)&d_matrix, tam_matriz));
	CUDA_CHK(cudaMalloc((void**)&d_sum_matrix, tam_matriz));

	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, tam_matriz, cudaMemcpyHostToDevice));

	
	dim3 tamGrid(ceil(FILAS / float(TAM_BLOQUE_X)), ceil(COLUMNAS / float(TAM_BLOQUE_Y)));
	dim3 tamBlock(TAM_BLOQUE_X, TAM_BLOQUE_Y);

	for (int i = 0; i < 10; i++) {
		sum_kernel << < tamGrid, tamBlock >> > (d_matrix, d_sum_matrix);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

		sum_kernel_optimizado << < tamGrid, tamBlock >> > (d_matrix, d_sum_matrix);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}
	
	
	CUDA_CHK(cudaMemcpy(h_matrix, d_sum_matrix, tam_matriz, cudaMemcpyDeviceToHost));

	//Para imprimir el resultado
	/*
	for (int i = 0; i < FILAS; i++) {
		for (int j = 0; j < COLUMNAS; j++) {
			cout << h_matrix[i * COLUMNAS + j] << " ";
		}
		cout << endl;
	}
	*/

	free(h_matrix);
	CUDA_CHK(cudaFree(d_matrix));
	CUDA_CHK(cudaFree(d_sum_matrix));

	return 0;
}


