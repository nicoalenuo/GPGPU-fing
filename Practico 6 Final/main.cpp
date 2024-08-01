#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

#include "CImg.h"

using namespace cimg_library;
using namespace std;

void filtro_mediana_cpu   (unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v1(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v2(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v3(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v4(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v5(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);
void filtro_mediana_gpu_v6(unsigned char* img_in, unsigned char* img_out, int width, int height, int W);

double calcular_promedio(vector<int> tiempos) {
    int suma = 0;
    for (int tiempo : tiempos) {
        suma += tiempo;
    }
    return suma / tiempos.size();
};

double calcular_desviacion_estandar(vector<int> tiempos, double media) {
    double suma = 0;
    for (int tiempo : tiempos) {
        suma += pow(tiempo - media, 2);
    }
    return sqrt(suma / (tiempos.size() - 1));
}


int main(int argc, char** argv) {
    chrono::high_resolution_clock::time_point marca_tiempo_anterior, marca_tiempo_actual;

    char* path = "imagenes_pre_procesamiento/2.pgm"; // Path por defecto
    int W = 5; // Tamaño de ventana por defecto

    if (argc < 2) { cout << "No se ingresó ningún path, se utilizará un archivo por defecto" << endl;} 
    else { path = argv[1]; }

    if (argc < 3) { cout << "No se ingresó el tamaño de ventana por pixel, se utilizará 5" << endl; } 
    else { W = stoi(argv[2]); }

    if (W % 2 == 0) { cout << "El tamaño de ventana debe ser impar, se utilizará " << W + 1 << endl; W++; }

    CImg<unsigned char> image(path);
    CImg<unsigned char> image_out(image.width(), image.height(), 1, 1, 0);

    unsigned char* img_matrix     = image.data();
    unsigned char* img_out_matrix = image_out.data();

    cout << "Width: "  << image.width()  << endl;
    cout << "Height: " << image.height() << endl;
    cout << "W: "      << W              << endl;
    cout << endl;

    vector<int> tiempos_cpu, tiempos_gpu_v1, tiempos_gpu_v2, tiempos_gpu_v3, tiempos_gpu_v4, tiempos_gpu_v5, tiempos_gpu_v6;
    int tiempo;

    cout << "Procesando imagen en CPU..." << endl;
    for (int i = 0; i < 11; i++) {
        marca_tiempo_anterior = chrono::high_resolution_clock::now();
        filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), W);
        marca_tiempo_actual = chrono::high_resolution_clock::now();
        tiempo = chrono::duration_cast<chrono::microseconds>(marca_tiempo_actual - marca_tiempo_anterior).count();
        cout << tiempo << "ms" << endl;
        if (i != 0) {
            tiempos_cpu.push_back(tiempo);
        }
        if (i == 10) {
            image_out.save("imagenes_post_procesamiento/output_cpu.pgm");
        }
    }

    for (int j = 1; j <= 6; j++) {
        cout << "Procesando imagen en GPU con version " << j << endl;
        vector<int> tiempos_gpu;
        for (int i = 0; i < 11; i++) {
            marca_tiempo_anterior = chrono::high_resolution_clock::now();
            switch (j) {
                case 1:
                    filtro_mediana_gpu_v1(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
                case 2:
                    filtro_mediana_gpu_v2(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
                case 3:
                    filtro_mediana_gpu_v3(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
                case 4:
                    filtro_mediana_gpu_v4(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
                case 5:
                    filtro_mediana_gpu_v5(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
                case 6:
                    filtro_mediana_gpu_v6(img_matrix, img_out_matrix, image.width(), image.height(), W);
                    break;
            }
            marca_tiempo_actual = chrono::high_resolution_clock::now();
            tiempo = chrono::duration_cast<chrono::microseconds>(marca_tiempo_actual - marca_tiempo_anterior).count();
            cout << tiempo << "ms" << endl;
            if (i != 0) {
                tiempos_gpu.push_back(tiempo);
            }
            if (i == 10) {
                switch (j) {
                    case 1:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v1.pgm");
                        tiempos_gpu_v1 = tiempos_gpu;
                        break;
                    case 2:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v2.pgm");
                        tiempos_gpu_v2 = tiempos_gpu;
                        break;
                    case 3:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v3.pgm");
                        tiempos_gpu_v3 = tiempos_gpu;
                        break;
                    case 4:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v4.pgm");
                        tiempos_gpu_v4 = tiempos_gpu;
                        break;
                    case 5:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v5.pgm");
                        tiempos_gpu_v5 = tiempos_gpu;
                        break;
                    case 6:
                        image_out.save("imagenes_post_procesamiento/output_gpu_v6.pgm");
                        tiempos_gpu_v6 = tiempos_gpu;
                        break;
                }
            }
        }
    }

    double promedio_cpu    = calcular_promedio(tiempos_cpu);
    double promedio_gpu_v1 = calcular_promedio(tiempos_gpu_v1);
    double promedio_gpu_v2 = calcular_promedio(tiempos_gpu_v2);
    double promedio_gpu_v3 = calcular_promedio(tiempos_gpu_v3);
    double promedio_gpu_v4 = calcular_promedio(tiempos_gpu_v4);
    double promedio_gpu_v5 = calcular_promedio(tiempos_gpu_v5);
    double promedio_gpu_v6 = calcular_promedio(tiempos_gpu_v6);

    cout << endl;
    cout << "Promedio de tiempo en 11 ejecuciones, descartando la primera:" << endl;
    cout << "CPU: "    << promedio_cpu    << " ms" << endl;
    cout << "GPU_v1: " << promedio_gpu_v1 << " ms" << endl;
    cout << "GPU_v2: " << promedio_gpu_v2 << " ms" << endl;
    cout << "GPU_v3: " << promedio_gpu_v3 << " ms" << endl;
    cout << "GPU_v4: " << promedio_gpu_v4 << " ms" << endl;
    cout << "GPU_v5: " << promedio_gpu_v5 << " ms" << endl;
    cout << "GPU_v6: " << promedio_gpu_v6 << " ms" << endl;

    cout << endl;
    cout << "Desviación estándar de tiempo en 11 ejecuciones, descartando la primera:" << endl;
    cout << "CPU: "    << calcular_desviacion_estandar(tiempos_cpu, promedio_cpu)       << " ms" << endl;
    cout << "GPU_v1: " << calcular_desviacion_estandar(tiempos_gpu_v1, promedio_gpu_v1) << " ms" << endl;
    cout << "GPU_v2: " << calcular_desviacion_estandar(tiempos_gpu_v2, promedio_gpu_v2) << " ms" << endl;
    cout << "GPU_v3: " << calcular_desviacion_estandar(tiempos_gpu_v3, promedio_gpu_v3) << " ms" << endl;
    cout << "GPU_v4: " << calcular_desviacion_estandar(tiempos_gpu_v4, promedio_gpu_v4) << " ms" << endl;
    cout << "GPU_v5: " << calcular_desviacion_estandar(tiempos_gpu_v5, promedio_gpu_v5) << " ms" << endl;
    cout << "GPU_v6: " << calcular_desviacion_estandar(tiempos_gpu_v6, promedio_gpu_v6) << " ms" << endl;

    return 0;
}
