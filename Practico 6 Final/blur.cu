#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Se definen diferentes tamaños de bloques
// El tamaño usado variara dependiendo del tamaño de ventana y version del kernel
// CUDA no admite mas de 48KB de memoria compartida por bloque en compute capabilites 6.0

#define TAM_BLOCK_X 32
#define TAM_BLOCK_Y 32

#define TAM_BLOCK_X_HALF 16
#define TAM_BLOCK_Y_HALF 16

#define TAM_BLOCK_X_QUARTER 8
#define TAM_BLOCK_Y_QUARTER 8

#define TAM_BLOCK_X_EIGHTH 4
#define TAM_BLOCK_Y_EIGHTH 4

using namespace std;

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Version simple, solo tiene acceso coalesed
__global__ 
void filtro_mediana_kernel_v1(unsigned char* img_in, unsigned char* img_out, unsigned char* aux_array, int width, int height, int W) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    int count = (i * width + j) * W * W; 
    const int inicio_aux_array = count;

    for (int i_aux = i - W / 2; i_aux <= i + W / 2; i_aux++) {
        for (int j_aux = j - W / 2; j_aux <= j + W / 2; j_aux++) {
            if (i_aux >= 0 && i_aux < height && j_aux >= 0 && j_aux < width) {
                aux_array[count] = img_in[i_aux * width + j_aux];
            } else {
                aux_array[count] = 0; 
            }
            count++;
        }
    }   

    unsigned char* temp_array = aux_array + inicio_aux_array;
    unsigned char temp;
    for (int i = 0; i < W * W; i++) {
        for (int j = i + 1; j < W * W; j++) {
            if (temp_array[i] > temp_array[j]) {
                temp = temp_array[i];
                temp_array[i] = temp_array[j];
                temp_array[j] = temp;
            }
        }
    }

    img_out[i * width + j] = temp_array[W * W / 2];
}

// Version igual a v1 con memoria compartida
__global__ 
void filtro_mediana_kernel_v2(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    extern __shared__ unsigned char shared_mem[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = W / 2;

    if (i >= width || j >= height) return;

    int id_local = threadIdx.y * blockDim.x + threadIdx.x;
    int xx, yy;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            xx = i + dx;
            yy = j + dy;
            shared_mem[id_local * W * W + (dy + radius) * W + (dx + radius)] = xx >= 0 && xx < width && yy >= 0 && yy < height ? img_in[yy * width + xx] : 0;
        }
    }

    unsigned char* temp_array = shared_mem + id_local * W * W;
    unsigned char temp;
    for (int k = 0; k < W * W; k++) {
        for (int l = k + 1; l < W * W; l++) {
            if (temp_array[k] > temp_array[l]) {
                temp = temp_array[k];
                temp_array[k] = temp_array[l];
                temp_array[l] = temp;
            }
        }
    }

    img_out[j * width + i] = temp_array[W * W / 2];
}


// Version que ordena el array usando radix sort
__global__ 
void filtro_mediana_kernel_v3(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    extern __shared__ unsigned char shared_mem[];

    unsigned char* elems     = shared_mem;         // Se guardarán los elementos de la ventana aqui
    unsigned char* elems_aux = shared_mem + W * W; // El resultado del reordenamiento parcial (respecto al bit n), se guardará aqui
    unsigned char* f         = shared_mem + 2 * W * W;
    unsigned char* aux; // Para hacer swap entre elems y elems_aux en lugar de copiar

    const int id_global = (blockIdx.y + threadIdx.y - (W / 2)) * width + (blockIdx.x + threadIdx.x - (W / 2));
    const int id_local  = threadIdx.y * blockDim.x + threadIdx.x;

    elems[id_local] = id_global >= 0 && id_global < width * height ? img_in[id_global] : 0;

    int total_falses, t, d;
    unsigned int b;

    // split(input, n)
    for (int i = 0; i < 8; i++){
        b = (elems[id_local] >> i) & 1;
        f[id_local] = !b;

        __syncthreads();

        if (id_local == 0){
            int old_val = f[0];
            int new_val;
            f[0] = 0;
            for (int i = 1; i < W * W; i++){
                new_val = f[i];
                f[i] = old_val + f[i - 1];
                old_val = new_val;
            }
        }
        
        __syncthreads();

        total_falses = f[W * W - 1] + !b;
        t = id_local - f[id_local] + total_falses;
        d = b ? t : f[id_local];

        elems_aux[d] = elems[id_local];

        aux = elems;
        elems = elems_aux;
        elems_aux = aux;
        
        __syncthreads();
    }

    if (id_local == 0){
        img_out[blockIdx.y * width + blockIdx.x] = elems[W * W / 2];
    }
}

//Cuenta cuantos elementos menores a pivot hay en el array, y los coloca a la izquierda (aunque no los ordena localmente)
__device__ 
int particion(unsigned char* window, int low, int high) {
    unsigned char pivot = window[high];
    int i = low - 1;
    unsigned char temp;
    for (int j = low; j < high; j++) {
        if (window[j] <= pivot) {
            i++;
            temp = window[i];
            window[i] = window[j];
            window[j] = temp;
        }
    }
    temp = window[i + 1];
    window[i + 1] = window[high];
    window[high] = temp;
    return i + 1;
}

// Obtiene el k-esimo elemento de un array
__device__ 
unsigned char quickselect(unsigned char* window, int low, int high, int k) {
    if (low <= high) {
        int pi = particion(window, low, high);
        if (pi == k)
            return window[pi];
        else if (pi < k)
            return quickselect(window, pi + 1, high, k);
        else
            return quickselect(window, low, pi - 1, k);
    }
    return 0;
}

// Utiliza el algoritmo quickselect para encontrar la mediana, en lugar de hacer sort del array
__global__ 
void filtro_mediana_kernel_v4(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height) return;

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ unsigned char shared_mem[];
    unsigned char* aux_array = shared_mem + idx * W * W;

    int radius = W / 2;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int xx = i + dx;
            int yy = j + dy;
            aux_array[(dy + radius) * W + (dx + radius)] = xx >= 0 && xx < width && yy >= 0 && yy < height ? img_in[yy * width + xx] : 0;
        }
    }

    img_out[j * width + i] = quickselect(aux_array, 0, W * W - 1, (W * W) / 2);
}


// Utiliza histogramas para calcular la mediana
__global__ 
void filtro_mediana_kernel_v5(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    extern __shared__ int shared_mem_2[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = (height / 64) * blockIdx.y;

    if (i >= width) return;

    int* histograma = shared_mem_2 + (256 + W) * threadIdx.x;
    int* row_to_delete = histograma + 256;
    
    for (int k = 0; k < 256; k++) {
        histograma[k] = 0;
    }

    const int radius = W / 2;
    int val, xx, yy;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            xx = i + dx;
            yy = j + dy;

            if (xx >= 0 && xx < width && yy >= 0 && yy < height){
                val = img_in[yy * width + xx];
            } else {
                val = 0;
            }
            histograma[val]++;  
            
            if (dy == -radius){
                row_to_delete[dx + radius] = val;
            }
        }
    }

    int elementos = 0;
    bool detenerse = false;
    for (int k = 0; !detenerse && k < 256; k++) {
        if (elementos + histograma[k] >= W * W / 2) {
            img_out[j * width + i] = k;
            detenerse = true;
        } else {
            elementos += histograma[k];
        }
    }
    
    for (int pos = 1; pos < height / 64; pos++){
        for (int k = 0; k < W; k++){ 
            histograma[row_to_delete[k]]--;

            xx = i - radius + k;
            yy = j + radius + pos;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height){
                histograma[img_in[yy * width + xx]]++;
            } else {
                histograma[0]++;
            }
            yy = j - radius + pos;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height){
                row_to_delete[k] = img_in[yy * width + xx];
            } else {
                row_to_delete[k] = 0;
            }
        }

        elementos = 0;
        detenerse = false;
        for (int k = 0; !detenerse && k < 256; k++) {
            if (elementos + histograma[k] >= W * W / 2) {
                img_out[(j + pos) * width + i] = k;
                detenerse = true;
            } else {
                elementos += histograma[k];
            }
        }
    }
}

// Version que usa el algoritmo de Torben para encontrar la mediana
__global__ 
void filtro_mediana_kernel_v6(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    extern __shared__ unsigned char shared_mem[];

    int i_g = blockIdx.x * blockDim.x + threadIdx.x;
    int j_g = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i_g >= width || j_g >= height) return;

    int i_l = threadIdx.x;
    int j_l = threadIdx.y;

    int half_W = W / 2;
    int shared_width = blockDim.x + W - 1;
    int shared_height = blockDim.y + W - 1;

    for (int j = threadIdx.y; j < shared_height; j += blockDim.y) {
        for (int i = threadIdx.x; i < shared_width; i += blockDim.x) {
            int x = min(max(blockIdx.x * blockDim.x + i - half_W, 0), width - 1);
            int y = min(max(blockIdx.y * blockDim.y + j - half_W, 0), height - 1);
            shared_mem[j * shared_width + i] = img_in[y * width + x];
        }
    }

    __syncthreads();

    unsigned char min = 255;
    unsigned char max = 0;
    
    for (int dy = -half_W; dy <= half_W; dy++) {
        for (int dx = -half_W; dx <= half_W; dx++) {
            unsigned char val = shared_mem[(j_l + half_W + dy) * (blockDim.x + W - 1) + i_l + half_W + dx];
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }

    unsigned char guess, maxltguess, mingtguess, val;
    int less, greater, equal;
    bool continuar = true;

    while (continuar) {
        guess = (min + max) / 2;
        less = greater = equal = 0;
        maxltguess = min;
        mingtguess = max;

        for (int dy = -half_W; dy <= half_W; dy++) {
            for (int dx = -half_W; dx <= half_W; dx++) {
                val = shared_mem[(j_l + half_W + dy) * (blockDim.x + W - 1) + i_l + half_W + dx];
                if (val < guess) {
                    less++;
                    if (val > maxltguess) maxltguess = val;
                } else if (val > guess) {
                    greater++;
                    if (val < mingtguess) mingtguess = val;
                } else {
                    equal++;
                }
            }
        }

        if (less <= W * W / 2 && greater <= W * W / 2) {
            continuar = false;
        } else if (less > greater) {
            max = maxltguess;
        } else {
            min = mingtguess;
        }
    }

    if (less <= W * W / 2) {
        img_out[j_g * width + i_g] = maxltguess;
    } else if (less + equal >= W * W / 2) {
        img_out[j_g * width + i_g] = guess;
    } else {
        img_out[j_g * width + i_g] = mingtguess;
    }
}

void filtro_mediana_gpu_v1(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;
    unsigned char* aux_array; // Memoria auxiliar que usará la v1 para ordenar

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&aux_array, width * height * W * W * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid(ceil(float(width) / TAM_BLOCK_X), ceil(float(height) / TAM_BLOCK_Y));
    dim3 dimBlock(TAM_BLOCK_X, TAM_BLOCK_Y);

    filtro_mediana_kernel_v1<<< dimGrid, dimBlock >>>(img_in_d, img_out_d, aux_array, width, height, W);
    CUDA_CHK( cudaGetLastError() );
    CUDA_CHK( cudaDeviceSynchronize() );

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    CUDA_CHK( cudaFree(aux_array) );
    // --------------------------
}

void filtro_mediana_gpu_v2(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid;
    dim3 dimBlock;
    size_t shared_memory_size;
    if (W < 7){
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X), ceil(float(height) / TAM_BLOCK_Y));
        dimBlock = dim3(TAM_BLOCK_X, TAM_BLOCK_Y);
        shared_memory_size = TAM_BLOCK_X * TAM_BLOCK_Y * W * W * sizeof(unsigned char);
    } else if (W < 15) {
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X_HALF), ceil(float(height) / TAM_BLOCK_Y_HALF));
        dimBlock = dim3(TAM_BLOCK_X_HALF, TAM_BLOCK_Y_HALF);
        shared_memory_size = TAM_BLOCK_X_HALF * TAM_BLOCK_Y_HALF * W * W * sizeof(unsigned char);
    } else {
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X_QUARTER), ceil(float(height) / TAM_BLOCK_Y_QUARTER));
        dimBlock = dim3(TAM_BLOCK_X_QUARTER, TAM_BLOCK_Y_QUARTER);
        shared_memory_size = TAM_BLOCK_X_QUARTER * TAM_BLOCK_Y_QUARTER * W * W * sizeof(unsigned char);
    }

    
    filtro_mediana_kernel_v2<< <dimGrid, dimBlock, shared_memory_size >>>(img_in_d, img_out_d, width, height, W);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    // --------------------------
}

void filtro_mediana_gpu_v3(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid(width, height);
    dim3 dimBlock(W, W);

    filtro_mediana_kernel_v3<<< dimGrid, dimBlock, W * W * 3 * sizeof(unsigned char) >>>(img_in_d, img_out_d, width, height, W);
    CUDA_CHK( cudaGetLastError() );
    CUDA_CHK( cudaDeviceSynchronize() );

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    // --------------------------
}

void filtro_mediana_gpu_v4(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid;
    dim3 dimBlock;
    size_t shared_memory_size;
    if (W < 7){
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X), ceil(float(height) / TAM_BLOCK_Y));
        dimBlock = dim3(TAM_BLOCK_X, TAM_BLOCK_Y);
        shared_memory_size = TAM_BLOCK_X * TAM_BLOCK_Y * W * W * sizeof(unsigned char);
    } else if (W < 15) {
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X_HALF), ceil(float(height) / TAM_BLOCK_Y_HALF));
        dimBlock = dim3(TAM_BLOCK_X_HALF, TAM_BLOCK_Y_HALF);
        shared_memory_size = TAM_BLOCK_X_HALF * TAM_BLOCK_Y_HALF * W * W * sizeof(unsigned char);
    } else {
        dimGrid = dim3(ceil(float(width) / TAM_BLOCK_X_QUARTER), ceil(float(height) / TAM_BLOCK_Y_QUARTER));
        dimBlock = dim3(TAM_BLOCK_X_QUARTER, TAM_BLOCK_Y_QUARTER);
        shared_memory_size = TAM_BLOCK_X_QUARTER * TAM_BLOCK_Y_QUARTER * W * W * sizeof(unsigned char);
    }

    filtro_mediana_kernel_v4 <<< dimGrid, dimBlock, shared_memory_size >>>(img_in_d, img_out_d, width, height, W);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    // --------------------------
}

void filtro_mediana_gpu_v5(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid(ceil(float(width) / TAM_BLOCK_X), 64);

    filtro_mediana_kernel_v5<<< dimGrid, TAM_BLOCK_X, TAM_BLOCK_X * (256 + W) * sizeof(int)>>>(img_in_d, img_out_d, width, height, W);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    // --------------------------
}

void filtro_mediana_gpu_v6(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    // Inicializacion de memoria ----------
    unsigned char* img_in_d;
    unsigned char* img_out_d;

    CUDA_CHK( cudaMalloc((void**)&img_in_d,  width * height * sizeof(unsigned char)) );
    CUDA_CHK( cudaMalloc((void**)&img_out_d, width * height * sizeof(unsigned char)) );

    CUDA_CHK( cudaMemcpy(img_in_d, img_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    //-------------------------------------

    dim3 dimGrid(ceil(float(width) / TAM_BLOCK_X), ceil(float(height) / TAM_BLOCK_Y));
    dim3 dimBlock(TAM_BLOCK_X, TAM_BLOCK_Y);

    filtro_mediana_kernel_v6<<< dimGrid, dimBlock, (TAM_BLOCK_X + W - 1) * (TAM_BLOCK_Y + W - 1) * sizeof(unsigned char) >>>(img_in_d, img_out_d, width, height, W);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copiar a memoria principal
    CUDA_CHK( cudaMemcpy(img_out, img_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    // Liberar memoria ----------
    CUDA_CHK( cudaFree(img_in_d) );
    CUDA_CHK( cudaFree(img_out_d) );
    // --------------------------
}

//-----------------------------------------------------
// Funciones para CPU ---------------------------------
//-----------------------------------------------------

// Implementación de quickselect en CPU
int partition_cpu(unsigned char* arr, int low, int high) {
    unsigned char pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            unsigned char temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    unsigned char temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

unsigned char quickselect_cpu(unsigned char* arr, int low, int high, int k) {
    if (low <= high) {
        int pi = partition_cpu(arr, low, high);
        if (pi == k)
            return arr[pi];
        else if (pi < k)
            return quickselect_cpu(arr, pi + 1, high, k);
        else
            return quickselect_cpu(arr, low, pi - 1, k);
    }
    return 0;
}

void filtro_mediana_cpu(unsigned char* img_in, unsigned char* img_out, int width, int height, int W) {
    unsigned char* colores = new unsigned char[W * W];

    int count = 0;
    // Recorro cada pixel
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            
            // Recorro la ventana alrededor del pixel
            for (int i_aux = i - W / 2; i_aux <= i + W / 2; i_aux++){
                for (int j_aux = j - W / 2; j_aux <= j + W / 2; j_aux++){
                    
                    if(i_aux >= 0 && i_aux < height && j_aux >= 0 && j_aux < width){
                        colores[count] = img_in[i_aux * width + j_aux];
                    } else {
                        colores[count] = 0;; 
                    }

                    count++;
                }
            }

            img_out[i * width + j] = quickselect_cpu(colores, 0, W * W - 1, W * W / 2);
            count = 0;
        }
    }

    delete[] colores;
}