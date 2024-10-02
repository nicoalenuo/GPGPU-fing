#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define FILAS 3840
#define COLUMNAS 2160
#define TAM_BLOQUE_X 16 //Debe dividir a COLUMNAS ('x' en los bloques de cuda se mueve entre las columnas)
#define TAM_BLOQUE_Y 32 //Debe dividir a FILAS ('y' en los bloques de cuda se mueve entre las filas)
#define VALOR_MAXIMO_MATRIZ 255 //La matriz contiene numeros de 0 a VALOR_MAXIMO_MATRIZ

#define TAM_BLOQUE_REDUCE 1024

#define CANT_BLOQUES_ADAPTADO 100 //Para el kernel adaptado del practico 2
#define CANT_HILOS_ADAPTADO 1024 //Para el kernel adaptado del practico 2

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void count_kernel_adaptado(int* d_matrix, int* d_histograma) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id; i < FILAS * COLUMNAS; i += gridDim.x * blockDim.x) {
		atomicAdd(&d_histograma[d_matrix[i]], 1);
	}
}

__global__ void count_kernel(int* d_matrix, int* d_histograma) {
	extern __shared__ int histograma_shared[];

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int id = COLUMNAS * y + x;

	histograma_shared[id % (VALOR_MAXIMO_MATRIZ + 1)] = 0; //seteo la memoria compartida en 0

	__syncthreads();

	atomicAdd(&histograma_shared[d_matrix[id]], 1);

	__syncthreads();

	if (threadIdx.x * threadIdx.y == 0) { //Solo un hilo de cada bloque hace la suma
		for (int i = 0; i <= VALOR_MAXIMO_MATRIZ; i++) {
			atomicAdd(&d_histograma[i], histograma_shared[i]);
		}
	}
}

__global__ void count_kernel_matrix(int* d_matrix, int* d_histograma_matrix) {
	extern __shared__ int histograma_shared[];

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int id = COLUMNAS * y + x;

	const int id_block = blockIdx.y * gridDim.x + blockIdx.x;

	histograma_shared[id % (VALOR_MAXIMO_MATRIZ + 1)] = 0;

	__syncthreads();

	atomicAdd(&histograma_shared[d_matrix[id]], 1);

	__syncthreads();

	d_histograma_matrix[(id_block * (VALOR_MAXIMO_MATRIZ + 1)) + (id % (VALOR_MAXIMO_MATRIZ + 1))] = histograma_shared[id % (VALOR_MAXIMO_MATRIZ + 1)];
}

__global__ void reduce_matrix(int* d_histograma_matrix, int desplazamiento) { 
    extern __shared__ int shared_block[];

    const int x = blockIdx.x;
    const int y = (blockDim.y * blockIdx.y + threadIdx.y) * desplazamiento;

    shared_block[threadIdx.y] = y < FILAS * COLUMNAS / (TAM_BLOQUE_X * TAM_BLOQUE_Y) ?
		d_histograma_matrix[(VALOR_MAXIMO_MATRIZ + 1) * y + x] :
		0;

    __syncthreads();

    for (int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (threadIdx.y < s) {
            shared_block[threadIdx.y] += shared_block[threadIdx.y + s];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        d_histograma_matrix[(VALOR_MAXIMO_MATRIZ + 1) * y + x] = shared_block[0];
    }
}


int main(int argc, char* argv[]) {
	int* h_matrix;
	int* d_matrix;

	int* h_histograma;
	int* d_histograma;

	int* h_histograma_matrix; //Matriz de histogramas
	int* d_histograma_matrix;

	//En bytes
	size_t tam_matrix = FILAS * COLUMNAS * sizeof(int);
	size_t tam_histograma = (VALOR_MAXIMO_MATRIZ + 1) * sizeof(int);
	size_t tam_histograma_matrix = (VALOR_MAXIMO_MATRIZ + 1) * (FILAS * COLUMNAS / (TAM_BLOQUE_X * TAM_BLOQUE_Y)) * sizeof(int);

	//Inicializacion de h_matrix
	//----------------------------
	h_matrix = (int*)malloc(tam_matrix);
	for (int i = 0; i < FILAS; i++) { //Al setear todos los valores en 1, el histograma tendrá todos los valores en 0, excepto por el 1 que valdra FILAS * COLUMNAS
		for (int j = 0; j < COLUMNAS; j++) {
			h_matrix[i * COLUMNAS + j] = 1;
		}
	}
	//----------------------------

	//Inicializacion de h_histograma
	//----------------------------
	h_histograma = (int*)malloc(tam_histograma);
	memset(h_histograma, 0, tam_histograma);
	//----------------------------

	//Inicializacion de h_histograma_matrix
	//----------------------------
	h_histograma_matrix = (int*)malloc(tam_histograma_matrix);
	memset(h_histograma_matrix, 0, tam_histograma_matrix);
	//----------------------------

	//Inicializacion de d_matrix
	//----------------------------
	CUDA_CHK(cudaMalloc((void**)&d_matrix, tam_matrix));
	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, tam_matrix, cudaMemcpyHostToDevice));
	//----------------------------
	
	//Inicializacion de d_histograma
	//----------------------------
	CUDA_CHK(cudaMalloc((void**)&d_histograma, tam_histograma));
	CUDA_CHK(cudaMemset(d_histograma, 0, tam_histograma));
	//----------------------------

	//Inicializacion de d_histograma_matrix
	//----------------------------
	CUDA_CHK(cudaMalloc((void**)&d_histograma_matrix, tam_histograma_matrix))
	CUDA_CHK(cudaMemset(d_histograma_matrix, 0, tam_histograma_matrix));
	//----------------------------

	dim3 tamGrid1(COLUMNAS / TAM_BLOQUE_X, FILAS / TAM_BLOQUE_Y);
	dim3 tamBlock1(TAM_BLOQUE_X, TAM_BLOQUE_Y);

	dim3 tamGrid2(VALOR_MAXIMO_MATRIZ + 1,  ceil(FILAS * COLUMNAS / (TAM_BLOQUE_X * TAM_BLOQUE_Y * float(TAM_BLOQUE_REDUCE))));
	dim3 tamBlock2(1, TAM_BLOQUE_REDUCE);

	dim3 tamGrid3(VALOR_MAXIMO_MATRIZ + 1,  ceil(FILAS * COLUMNAS / (TAM_BLOQUE_X * TAM_BLOQUE_Y * TAM_BLOQUE_REDUCE * float(TAM_BLOQUE_REDUCE))));
	dim3 tamBlock3(1, TAM_BLOQUE_REDUCE);

	//for (int i = 0; i < 10; i++) {

		count_kernel_adaptado <<< CANT_BLOQUES_ADAPTADO, CANT_HILOS_ADAPTADO >>> (d_matrix, d_histograma);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
		
		count_kernel <<< tamGrid1, tamBlock1, tam_histograma >>> (d_matrix, d_histograma);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
		
		count_kernel_matrix <<< tamGrid1, tamBlock1, tam_histograma >>> (d_matrix, d_histograma_matrix);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

		reduce_matrix <<< tamGrid2, tamBlock2, TAM_BLOQUE_REDUCE * sizeof(int) >>> (d_histograma_matrix, 1);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

		reduce_matrix <<< tamGrid3, tamBlock3, TAM_BLOQUE_REDUCE * sizeof(int) >>> (d_histograma_matrix, TAM_BLOQUE_REDUCE);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());

	//}

	CUDA_CHK(cudaMemcpy(h_histograma, d_histograma, tam_histograma, cudaMemcpyDeviceToHost));
	CUDA_CHK(cudaMemcpy(h_histograma_matrix, d_histograma_matrix, tam_histograma_matrix, cudaMemcpyDeviceToHost));

	for (int i = 0; i <= VALOR_MAXIMO_MATRIZ; i++) {
		cout << i << " : " << h_histograma_matrix[i] << endl;
	}

	free(h_matrix);
	free(h_histograma);
	free(h_histograma_matrix);
	CUDA_CHK(cudaFree(d_matrix));
	CUDA_CHK(cudaFree(d_histograma));
	CUDA_CHK(cudaFree(d_histograma_matrix));

	return 0;
}