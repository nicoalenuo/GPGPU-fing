#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>
#include <cub/cub.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define N 1
#define TAM_ARRAY 1024 * (1 << N)
#define TAM_BLOQUES 1024

using namespace std;

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void exclusive_scan_kernel(int* d_array, int* d_array_resultado, int* d_resultado_final_por_bloque, int n) {
    extern __shared__ int shared_block[]; 

    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    shared_block[threadIdx.x] = (indice < n) ? d_array[indice] : 0;

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int posicion_sumando = offset * (2 * threadIdx.x + 1) - 1;
        int posicion_suma = offset * (2 * threadIdx.x + 2) - 1;
        if (posicion_suma < blockDim.x) {
            shared_block[posicion_suma] += shared_block[posicion_sumando];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_resultado_final_por_bloque[blockIdx.x] = shared_block[blockDim.x - 1];
        shared_block[blockDim.x - 1] = 0;
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        int posicion_sumando = offset * (2 * threadIdx.x + 1) - 1;
        int posicion_suma = offset * (2 * threadIdx.x + 2) - 1;
        if (posicion_suma < blockDim.x) {
            int temporal = shared_block[posicion_sumando];
            shared_block[posicion_sumando] = shared_block[posicion_suma];
            shared_block[posicion_suma] += temporal;
        }
        __syncthreads();
    }

    if (indice < n) {
        d_array_resultado[indice] = shared_block[threadIdx.x];
    }
}


__global__ void sumar_resultados_finales_por_bloque(int* d_resultado_final_por_bloque, int* d_array_resultado) {
    d_array_resultado[blockIdx.x * blockDim.x + threadIdx.x] += d_resultado_final_por_bloque[blockIdx.x];
}

void scan(int* d_array, int* d_array_resultado, int tam_array){
    int cant_bloques = ceil(float(tam_array) / TAM_BLOQUES);

    int* d_resultado_final_por_bloque;
    int* d_resultado_final_por_bloque_scan;
    CUDA_CHK(cudaMalloc((void**)&d_resultado_final_por_bloque, cant_bloques * sizeof(int)));
    CUDA_CHK(cudaMalloc((void**)&d_resultado_final_por_bloque_scan, cant_bloques * sizeof(int)));

    exclusive_scan_kernel<<<cant_bloques, TAM_BLOQUES, TAM_BLOQUES * sizeof(int)>>>(d_array, d_array_resultado, d_resultado_final_por_bloque, tam_array);
    CUDA_CHK(cudaGetLastError());
	CUDA_CHK(cudaDeviceSynchronize());

    if (cant_bloques > 1){
        scan(d_resultado_final_por_bloque, d_resultado_final_por_bloque_scan, cant_bloques); //Llamada recursiva para hacer scan de los resultados intermedios entre bloques

        sumar_resultados_finales_por_bloque<<<cant_bloques, TAM_BLOQUES>>>(d_resultado_final_por_bloque_scan, d_array_resultado);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
    }
    
    CUDA_CHK(cudaFree(d_resultado_final_por_bloque));
    CUDA_CHK(cudaFree(d_resultado_final_por_bloque_scan));
}

int main(int argc, char* argv[]) {
	int* h_array;
	int* d_array;

    int* h_array_resultado;
    int* d_array_resultado;

    thrust::host_vector<int> h_array_thrust(TAM_ARRAY);
    thrust::device_vector<int> d_array_thrust(TAM_ARRAY);
    thrust::host_vector<int> h_array_resultado_thrust(TAM_ARRAY, 0);
    thrust::device_vector<int> d_array_resultado_thrust(TAM_ARRAY, 0);

    int* h_array_resultado_cub;
    int* d_array_resultado_cub;
    int* d_almacentamiento_temporal_cub = nullptr;

	//En bytes
	size_t tam_array = TAM_ARRAY * sizeof(int);
    size_t tam_almacentamiento_temporal_cub = 0;

	//Inicializacion de h_array
	//----------------------------
	h_array = (int*)malloc(tam_array);
    for (int i = 0; i < TAM_ARRAY; i++) {
        h_array[i] = i;
    }
	//----------------------------
    
	//Inicializacion de d_array
	//----------------------------
    CUDA_CHK(cudaMalloc((void**)&d_array, tam_array));
    CUDA_CHK(cudaMemcpy(d_array, h_array, tam_array, cudaMemcpyHostToDevice));
	//----------------------------

	//Inicializacion de h_array_resultado
	//----------------------------
	h_array_resultado = (int*)malloc(tam_array);
    memset(h_array_resultado, 0, tam_array);
	//----------------------------

	//Inicializacion de d_array_resultado
	//----------------------------
    CUDA_CHK(cudaMalloc((void**)&d_array_resultado, tam_array));
    CUDA_CHK(cudaMemcpy(d_array_resultado, h_array_resultado, tam_array, cudaMemcpyHostToDevice));
	//----------------------------

    //Inicializacion de h_array_thrust
    //----------------------------
    for (int i = 0; i < TAM_ARRAY; i++) {
        h_array_thrust[i] = h_array[i];
    }
    //----------------------------

    //Inicializacion de d_array_thrust
    //----------------------------
    d_array_thrust = h_array_thrust;
    //----------------------------

    //Inicializacion de h_array_resultado_cub
    //----------------------------
    h_array_resultado_cub = (int*)malloc(tam_array);
    memset(h_array_resultado_cub, 0, tam_array);
    //----------------------------

    //Inicializacion de d_array_resultado_cub
    //----------------------------
    CUDA_CHK(cudaMalloc((void**)&d_array_resultado_cub, tam_array));
    CUDA_CHK(cudaMemcpy(d_array_resultado_cub, h_array_resultado_cub, tam_array, cudaMemcpyHostToDevice));
    //----------------------------

    //Inicializacion de d_almacenamiento_temporal_cub
    //----------------------------
    cub::DeviceScan::ExclusiveSum(d_almacentamiento_temporal_cub, tam_almacentamiento_temporal_cub, d_array, d_array_resultado_cub, TAM_ARRAY);
    CUDA_CHK(cudaMalloc((void**)&d_almacentamiento_temporal_cub, tam_almacentamiento_temporal_cub));
    //----------------------------
    
    chrono::high_resolution_clock::time_point marca_tiempo_anterior, marca_tiempo_actual;

    vector<int> tiempos_scan;
    vector<int> tiempos_cub;
    vector<int> tiempos_thrust;

    for (int i = 0; i < 11; i++) {
        marca_tiempo_anterior = chrono::high_resolution_clock::now();
		scan(d_array, d_array_resultado, TAM_ARRAY);
        marca_tiempo_actual = chrono::high_resolution_clock::now();
        tiempos_scan.push_back(chrono::duration_cast<chrono::microseconds>(marca_tiempo_actual - marca_tiempo_anterior).count());

        marca_tiempo_anterior = chrono::high_resolution_clock::now();
        cub::DeviceScan::ExclusiveSum(d_almacentamiento_temporal_cub, tam_almacentamiento_temporal_cub, d_array, d_array_resultado_cub, TAM_ARRAY);
        marca_tiempo_actual = chrono::high_resolution_clock::now();
        tiempos_cub.push_back(chrono::duration_cast<chrono::microseconds>(marca_tiempo_actual - marca_tiempo_anterior).count());

        marca_tiempo_anterior = chrono::high_resolution_clock::now();
        thrust::exclusive_scan(d_array_thrust.begin(), d_array_thrust.end(), d_array_resultado_thrust.begin());
        marca_tiempo_actual = chrono::high_resolution_clock::now();
        tiempos_thrust.push_back(chrono::duration_cast<chrono::microseconds>(marca_tiempo_actual - marca_tiempo_anterior).count());
	}
    
    tiempos_scan.erase(tiempos_scan.begin());
    tiempos_cub.erase(tiempos_cub.begin());
    tiempos_thrust.erase(tiempos_thrust.begin());

    float tiempo_total_scan = 0;
    float tiempo_total_cub = 0;
    float tiempo_total_thrust = 0;

    for (int i = 0; i < 11; i++) {
        tiempo_total_scan += tiempos_scan[i];
        tiempo_total_cub += tiempos_cub[i];
        tiempo_total_thrust += tiempos_thrust[i];
    }
    cout << "Tiempos de ejecucion de Scan (microsegundos):" << endl;
    for (int i = 0; i < 11; i++) {
        cout << tiempos_scan[i] << endl;
    }
    cout << "Tiempo promedio de ejecucion de scan: " << tiempo_total_scan / 10 << " microsegundos" << endl;

    cout << "Tiempos de ejecucion de cub:" << endl;
    for (int i = 0; i < 11; i++) {
        cout <<tiempos_cub[i] << endl;
    }
    cout << "Tiempo promedio de ejecucion de cub: " << tiempo_total_cub / 10 << " microsegundos" << endl;

    cout << "Tiempos de ejecucion de thrust:" << endl;

    for (int i = 0; i < 11; i++) {
        cout << tiempos_thrust[i] << endl;
    }
    cout << "Tiempo promedio de ejecucion de thrust: " << tiempo_total_thrust / 10 << " microsegundos" << endl;



    CUDA_CHK(cudaMemcpy(h_array_resultado, d_array_resultado, tam_array, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(h_array_resultado_cub, d_array_resultado_cub, tam_array, cudaMemcpyDeviceToHost));
    h_array_resultado_thrust = d_array_resultado_thrust;

    // Comparar los tres resultados
    for (int i = 0; i < TAM_ARRAY; i++) {
        if (h_array_resultado[i] != h_array_resultado_cub[i] || h_array_resultado[i] != h_array_resultado_thrust[i] || h_array_resultado_cub[i] != h_array_resultado_thrust[i]) {
            cerr << "Error en la posicion " << i << endl;
            cerr << "Resultado manual: " << h_array_resultado[i] << endl;
            cerr << "Resultado cub: " << h_array_resultado_cub[i] << endl;
            cerr << "Resultado thrust: " << h_array_resultado_thrust[i] << endl;
            cerr << endl;
            exit(1);
        }
    }


    free(h_array);
    free(h_array_resultado);

    CUDA_CHK(cudaFree(d_array));
    CUDA_CHK(cudaFree(d_array_resultado));

    free(h_array_resultado_cub);
    CUDA_CHK(cudaFree(d_array_resultado_cub));
    CUDA_CHK(cudaFree(d_almacentamiento_temporal_cub));

    //Thrust se encarga de liberar la memoria de los vectores

	return 0;
}