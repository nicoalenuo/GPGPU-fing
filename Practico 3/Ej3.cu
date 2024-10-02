#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>

using namespace std;

#define FILAS 10240
#define COLUMNAS 256

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void mult_kernel(int* d_matrix, int* d_vector, int* d_vector_resultado) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int resultado = 0;

	for (int i = 0; i < COLUMNAS; i++) 
		resultado += d_matrix[id * COLUMNAS + i] * d_vector[i];

	d_vector_resultado[id] = resultado;
}

__global__ void mult_kernel_optimizado(int* d_matrix, int* d_vector, int* d_vector_resultado) {
	extern __shared__ int sumas_parciales[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	sumas_parciales[threadIdx.x] = d_matrix[id] * d_vector[threadIdx.x];

	__syncthreads();

	if (threadIdx.x == 0) {
		int resultado = 0;
		for (int i = 0; i < COLUMNAS; i++)
			resultado += sumas_parciales[i];

		d_vector_resultado[blockIdx.x] = resultado;
	}
}

__global__ void mult_kernel_optimizado_reduce(int* d_matrix, int* d_vector, int* d_vector_resultado) {
	extern __shared__ int sumas_parciales[];
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	sumas_parciales[threadIdx.x] = d_matrix[id] * d_vector[threadIdx.x];

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			sumas_parciales[threadIdx.x] = sumas_parciales[threadIdx.x] + sumas_parciales[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	if (threadIdx.x == 0)
		d_vector_resultado[blockIdx.x] = sumas_parciales[0];

}

int main(int argc, char* argv[]) {
	size_t tam_matrix = FILAS * COLUMNAS * sizeof(int);
	size_t tam_vector = COLUMNAS * sizeof(int);
	size_t tam_vector_resultado = FILAS * sizeof(int);

	int* h_matrix;
	int* h_vector;
	int* h_vector_resultado;

	int* d_matrix;
	int* d_vector;
	int* d_vector_resultado;

	h_matrix = (int*)malloc(tam_matrix);
	h_vector = (int*)malloc(tam_vector);
	h_vector_resultado = (int*)malloc(tam_vector_resultado);

	for (int i = 0; i < FILAS; i++) 
		for (int j = 0; j < COLUMNAS; j++) 
			h_matrix[i * COLUMNAS + j] = 1;

	for (int j = 0; j < COLUMNAS ; j++)
		h_vector[j] = 1;

	CUDA_CHK(cudaMalloc((void**)&d_matrix, tam_matrix));
	CUDA_CHK(cudaMalloc((void**)&d_vector, tam_vector));
	CUDA_CHK(cudaMalloc((void**)&d_vector_resultado, tam_vector_resultado));

	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, tam_matrix, cudaMemcpyHostToDevice));
	CUDA_CHK(cudaMemcpy(d_vector, h_vector, tam_vector, cudaMemcpyHostToDevice));

	for (int i = 0; i < 10; i++) {
		mult_kernel << <FILAS / 1024, 1024 >> > (d_matrix, d_vector, d_vector_resultado);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

		mult_kernel_optimizado << <FILAS, COLUMNAS, COLUMNAS * sizeof(int) >> > (d_matrix, d_vector, d_vector_resultado);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

		mult_kernel_optimizado_reduce << <FILAS, COLUMNAS, COLUMNAS * sizeof(int) >> > (d_matrix, d_vector, d_vector_resultado);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}

	CUDA_CHK(cudaMemcpy(h_vector_resultado, d_vector_resultado, tam_vector_resultado, cudaMemcpyDeviceToHost));

	for (int i = 0; i < FILAS; i++)
		cout << h_vector_resultado[i] << endl;

	free(h_matrix);
	free(h_vector);
	free(h_vector_resultado);
	CUDA_CHK(cudaFree(d_matrix));
	CUDA_CHK(cudaFree(d_vector));
	CUDA_CHK(cudaFree(d_vector_resultado));

	return 0;
}


