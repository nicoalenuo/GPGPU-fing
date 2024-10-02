#include <stdio.h>
#include <x86intrin.h>
#include <iostream>

const size_t SIZE = 1024*8;
const size_t CACHE_L1_SIZE = 1024*32;
const size_t CACHE_L2_SIZE = 1024*256;

int main() {
    char a = 'a';

    int sig_indice;

    unsigned long long ciclosL1 = 0;
    unsigned long long ciclosL2 = 0;
    unsigned long long ciclosL3 = 0;
    unsigned long long start = 0;
    unsigned long long end = 0;

    char* data = new char[SIZE];
    char* rellenoL1 = new char[CACHE_L1_SIZE];
    char* rellenoL2 = new char[CACHE_L2_SIZE];

    for (int k = 0; k < CACHE_L1_SIZE ; k++){
        rellenoL1[k] = 'a';
    }
    for (int k = 0; k < CACHE_L2_SIZE ; k++){
        rellenoL2[k] = 'b';
    }
    for (int i = 0; i < SIZE ; i++) {
        data[i] = 'c';
    }
    for (int k = 0; k<16384; k++){

        for (int i = 0; i < SIZE ; i++) { //Guardar array en cache L1
            a += data[i];
        }

        sig_indice = rand() % SIZE;

        start = __rdtsc();
        a = data[sig_indice];
        end = __rdtsc();

        ciclosL1 += end - start;

        for (int i = 0; i < CACHE_L1_SIZE ; i++) { //Llenar cache L1, desplazando al array a cache L2
            a += rellenoL1[i];
        }

        sig_indice = rand() % SIZE;

        start = __rdtsc();
        a = data[sig_indice];
        end = __rdtsc();

        ciclosL2 += end - start;
        for (int i = 0; i < CACHE_L2_SIZE ; i++) { //Llenar cache L2, desplazando al array a cache L3
            a += rellenoL2[i];
        }

        sig_indice = rand() % SIZE;

        start = __rdtsc();
        a = data[sig_indice];
        end = __rdtsc();

        ciclosL3 += end - start;
    }
    printf("Ciclos promedio por lectura en L1: %i ciclos\n", ciclosL1/16384);
    printf("Ciclos promedio por lectura en L2: %i ciclos\n", ciclosL2/16384);
    printf("Ciclos promedio por lectura en L3: %i ciclos\n", ciclosL3/16384);
    return 0;
}
