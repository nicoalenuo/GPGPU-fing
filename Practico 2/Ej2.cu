#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
  
#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define CANT_BLOQUES 5
#define CANT_HILOS 1024

#define N 512

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void read_file(const char*, int*);
int get_text_length(const char* fname);

__device__ int modulo(int a, int b) {
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int* d_message, int length) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id; i < length; i += gridDim.x * blockDim.x) {
		char letraEncriptada = d_message[i];
		char letraDesencriptada = modulo(A_MMI_M * (letraEncriptada - B), M);
		d_message[i] = letraDesencriptada;
	}
}

__global__ void count_kernel(int* d_message, int* d_counter, int length) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id; i < length; i += gridDim.x * blockDim.x) {
		atomicAdd(&d_counter[d_message[i]], 1); 
	}
}



int* count_letters(int* d_message, int length) {
	int* h_counter;
	int* d_counter;

	h_counter = (int*)malloc(256 * sizeof(int));

	CUDA_CHK(cudaMalloc((void**)&d_counter, 256 * sizeof(int)));
	CUDA_CHK(cudaMemset(d_counter, 0, 256 * sizeof(int)));

	dim3 tamGrid(CANT_BLOQUES, 1);
	dim3 tamBlock(CANT_HILOS, 1, 1);

	count_kernel << <tamGrid, tamBlock >> > (d_message, d_counter, length);
	CUDA_CHK(cudaGetLastError());
	CUDA_CHK(cudaDeviceSynchronize());

	CUDA_CHK(cudaMemcpy(h_counter, d_counter, 256 * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHK(cudaFree(d_counter));

	return h_counter;
}

int main(int argc, char* argv[]) {
	int* h_message;
	int* d_message;

	unsigned int size;

	const char* fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);
	size = length * sizeof(int);

	h_message = (int*)malloc(size);
	read_file(fname, h_message);

	CUDA_CHK(cudaMalloc((void**)&d_message, size));
	CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

	dim3 tamGrid(CANT_BLOQUES, 1);
	dim3 tamBlock(CANT_HILOS, 1, 1);

	decrypt_kernel <<<tamGrid, tamBlock >>> (d_message, length);
	CUDA_CHK(cudaGetLastError());
	CUDA_CHK(cudaDeviceSynchronize());


	CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

	int* h_counter = count_letters(d_message, length);

	for (int i = 0; i < 256; i++) {
		printf("%c: %i ", char(i), h_counter[i]);
	}
	printf("\n");

	CUDA_CHK(cudaFree(d_message));
	free(h_message);
	free(h_counter);

	return 0;
}


int get_text_length(const char* fname)
{
	FILE* f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);
	fseek(f, 0, SEEK_END);
	size_t length = ftell(f);
	fseek(f, pos, SEEK_SET);

	fclose(f);

	return length;
}

void read_file(const char* fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE* f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c;
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
