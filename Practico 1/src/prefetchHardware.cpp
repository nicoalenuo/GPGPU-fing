#include <iostream>
#include <time.h>
#include <string.h>
using namespace std;
const long SECONDS_PER_NS = 1000000000;
const size_t tam_bloque = 64;
const size_t tam_array = (1<<26) * tam_bloque; //Array de 4GB
struct timespec comienzoTimer, finTimer;
double tiempo_transcurrido;


char a; //Variable donde se guardarán los valores leídos

void acceder_sin_prefetch_hardware(char* inicio_array, size_t tam_bloque, size_t tam_array){

    char* final_array = inicio_array + tam_array;
    char* iterador = inicio_array; 
    
    //Acceso a bloques pares
    for (iterador = inicio_array; iterador < final_array; iterador += 2*tam_bloque)
        for (int i = 0; i < tam_bloque ;  i++)
            a = *(iterador + i);

    //Acceso a bloques impares
    for (iterador = inicio_array + tam_bloque; iterador < final_array; iterador += 2*tam_bloque)
        for (int i = 0; i < tam_bloque ;  i++)
            a = *(iterador + i);

}

void acceder_con_prefetch_hardware(char* inicio_array, size_t tam_bloque, size_t tam_array){

    char* final_array = inicio_array + tam_array;
    char* iterador = inicio_array; 
    
    for (int i = 0; i < tam_array ; i++)
        a = *(iterador + i);

}

int main(){

    char* array_inicio = new char[tam_array]; 
    memset(array_inicio, 'a', tam_array); // Lleno el array con 'a'

    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
    acceder_sin_prefetch_hardware(array_inicio, tam_bloque, tam_array);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;

    cout << "Tiempo sin prefetch: " << tiempo_transcurrido << " segundos" << endl;

    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
    acceder_con_prefetch_hardware(array_inicio, tam_bloque, tam_array);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;
    cout << "Tiempo con prefetch: " << tiempo_transcurrido << " segundos" << endl;


    delete[] array_inicio;
}