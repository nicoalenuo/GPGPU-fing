#include <iostream>
#include <emmintrin.h>
#include <time.h>
#include <string.h>

using namespace std;
const size_t tam_bloque = 64;
const size_t tam_fila = (1<<7)*tam_bloque;
const long SECONDS_PER_NS = 1000000000;
struct timespec comienzoTimer, finTimer;

double tiempo_transcurrido;
char a;

void acceder_sin_prefetch_software(char* matriz_inicio, size_t tam_bloque, size_t tam_matriz, size_t tam_fila){
    char* matriz_final = matriz_inicio + tam_matriz;

    char* iterador = matriz_inicio; //El iterador será desde donde comenzarán a moverse los dos siguientes punteros, se moverá sobre la diagonal de la matriz
    char* puntero_x; //Puntero_x se movera bloque a bloque sobre el eje x
    char* puntero_y; //Puntero_y se movera bloque a bloque sobre el eje y
    
    while (iterador < matriz_final - tam_fila){ //Mientras no llegue al ultimo bloque en la diagonal
        puntero_x = iterador + tam_bloque; 
        puntero_y = iterador + tam_fila; 

        while (puntero_y < matriz_final){
            for (int i = 0; i < tam_bloque ; i++){ //Accedo a los dos siguientes bloques
                a = *(puntero_x + i); 
                a = *(puntero_y + i);
            }

            puntero_x += tam_bloque; //Avanzo al siguiente bloque en x
            puntero_y += tam_fila; //Avanzo al siguiente bloque en y
        }

        iterador += tam_fila + tam_bloque; //Avanzo al siguiente bloque en la diagonal
    }
}


//El codigo es el mismo que el anterior, solo se agregan las dos instrucciones para prefetch
void acceder_con_prefetch_software(char* matriz_inicio, size_t tam_bloque, size_t tam_matriz, size_t tam_fila){
    char* matriz_final = matriz_inicio + tam_matriz;

    char* iterador = matriz_inicio; 
    char* puntero_x; 
    char* puntero_y;
    
    while (iterador < matriz_final - tam_fila){ 
        puntero_x = iterador + tam_bloque; 
        puntero_y = iterador + tam_fila; 

        while (puntero_y < matriz_final){
            
            _mm_prefetch((char*)(puntero_x + tam_bloque), _MM_HINT_T0); //Hago prefetch del siguiente bloque en x
            _mm_prefetch((char*)(puntero_y + tam_fila), _MM_HINT_T0); //y el siguiente bloque en y, estos se usaran en la proxima iteracion

            for (int i = 0; i < tam_bloque ; i++){ 
                a = *(puntero_x + i); 
                a = *(puntero_y + i);
            }

            puntero_x += tam_bloque; 
            puntero_y += tam_fila; 
        }

        iterador += tam_fila + tam_bloque; 
    }
}

int main(){
    size_t tam_matriz = tam_fila * tam_fila; // Matriz de 64MB

    char* matriz_inicio = new char[tam_matriz]; 
    memset(matriz_inicio, 'a', tam_matriz); // Lleno la matriz con 'a'

    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
    acceder_sin_prefetch_software(matriz_inicio, tam_bloque, tam_matriz, tam_fila);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;

    cout << "Tiempo sin prefetch: " << tiempo_transcurrido << " segundos" << endl;

    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
    acceder_con_prefetch_software(matriz_inicio, tam_bloque, tam_matriz, tam_fila);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;

    cout << "Tiempo con prefetch: " << tiempo_transcurrido << " segundos" << endl;

    delete[] matriz_inicio;
}