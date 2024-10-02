#include <iostream>
#include <ctime>

using namespace std;

const size_t ARRAY_SIZE = 2048;
const size_t CANT_ITER = 1000000;
struct timespec comienzoTimer, finTimer;

struct estructura_desalineada {
    float f1;
    float f2;
    float f3;
    float f4;
    float f5;
    float f6;
    float f7;
    float f8;
    float f9;
    float f10;
    float f11;
    float f12;
};

struct alignas(64) estructura_alineada {
    float f1;
    float f2;
    float f3;
    float f4;
    float f5;
    float f6;
    float f7;
    float f8;
    float f9;
    float f10;
    float f11;
    float f12;
};

float f;

void ejecutar_desalineada(estructura_desalineada array[]) {
    for (int i = 0; i < CANT_ITER; i++) {
        for (int k = 0; k < ARRAY_SIZE; k++) {
            f = array[k].f1;
            f = array[k].f2;
            f = array[k].f3;
            f = array[k].f4;
            f = array[k].f5;
            f = array[k].f6;
            f = array[k].f7;
            f = array[k].f8;
            f = array[k].f9;
            f = array[k].f10;
            f = array[k].f11;
            f = array[k].f12;
        }
    }
}

void ejecutar_alineada(estructura_alineada array[]) {
    for (int i = 0; i < CANT_ITER; i++) {
        for (int k = 0; k < ARRAY_SIZE; k++) {
            f = array[k].f1;
            f = array[k].f2;
            f = array[k].f3;
            f = array[k].f4;
            f = array[k].f5;
            f = array[k].f6;
            f = array[k].f7;
            f = array[k].f8;
            f = array[k].f9;
            f = array[k].f10;
            f = array[k].f11;
            f = array[k].f12;
        }
    }
}

int main() {
    estructura_desalineada* array_desalineado = new estructura_desalineada[ARRAY_SIZE];
    estructura_alineada* array_alineado = new estructura_alineada[ARRAY_SIZE];

    double tiempo_transcurrido;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &comienzoTimer);
    ejecutar_desalineada(array_desalineado);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;

    cout << "Duracion al usar una estructura desalineada: " << tiempo_transcurrido << " s" << endl;


    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &comienzoTimer);
    ejecutar_alineada(array_alineado);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finTimer);

    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;

    cout << "Duracion al usar una estructura alineada: " << tiempo_transcurrido << " s" << endl;

    delete[] array_desalineado;
    delete[] array_alineado;
}