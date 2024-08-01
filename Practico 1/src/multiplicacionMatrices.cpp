#include <stdio.h> 
#include <time.h>
#include <string.h>
#include <malloc.h>
#include <cstdlib>

struct timespec comienzoTimer, finTimer;

const int BLOCK_SIZEm = 64;
const int BLOCK_SIZEn = 64;
const int BLOCK_SIZEp = 64;

const size_t m = 1024;
const size_t p = 1024;
const size_t n = 1024;

void reordenamiento_row_col_it(float A[], float B[], float C[]){
    float sum = 0;
    for (int row = 0; row < m; row++){
        for (int col = 0; col < n; col++){
            sum=0;
            for (int it = 0; it < p; it++){
                sum += A[row * p + it] * B[it * n + col];
                C[row * n + col]=sum;

            }
        }
    }
}

void reordenamiento_col_row_it(float A[], float B[], float C[]){
    float sum = 0;
    for (int col = 0; col < n; col++){
        for (int row = 0; row < m; row++){
            sum=0;
            for (int it = 0; it < p; it++){
                sum += A[row * p + it] * B[it * n + col];
                C[row * n + col] = sum;
            }
        }
    }
}

void reordenamiento_it_col_row(float A[], float B[], float C[]){
    float sum = 0;
    for (int it = 0; it < p; it++){
        for (int col = 0; col < n; col++){
            sum=0;
            for (int row = 0; row < m; row++){
                sum = A[row * p + it] * B[it * n + col];
                C[row * n + col]+=sum;
            }
        }
    }
}

void reordenamiento_it_row_col(float A[], float B[], float C[]){
    float sum = 0;
    for (int it = 0; it < p; it++){
        for (int row = 0; row < m; row++){
            sum=0;
            for (int col = 0; col < n; col++){
                sum = A[row * p + it] * B[it * n + col];
                C[row * n + col] += sum;
            }
        }
    }
}

void reordenamiento_row_it_col(float A[], float B[], float C[]){
    float sum = 0;
    for (int row = 0; row < m; row++){
        for (int it = 0; it < p; it++){
            sum=0;
            for (int col = 0; col < n; col++){
                sum = A[row * p + it] * B[it * n + col];
                C[row * n + col] += sum;
            }
        }
    }
}

void reordenamiento_col_it_row(float A[], float B[], float C[]){
    float sum = 0;
    for (int col = 0; col < n; col++){
        for (int it = 0; it < p; it++){
        sum=0;
            for (int row = 0; row < m; row++){
                sum = A[row * p + it] * B[it * n + col];
                C[row * n + col] +=sum;

            }
        }
    }
}

void reordenamiento_row_col_it_con_blocking(float A[], float B[], float C[]){
    float sum = 0;

    for (int row = 0; row < m; row += BLOCK_SIZEm) {
        for (int col = 0; col < n; col += BLOCK_SIZEn) {
            for (int it = 0; it < p; it += BLOCK_SIZEp) {

                for (int ii = row; ii < row + BLOCK_SIZEm; ii++) {
                    for (int jj = col; jj < col + BLOCK_SIZEn; jj++) {
                        sum = 0;
                        for (int kk = it; kk < it + BLOCK_SIZEp; kk++) {
                            sum = A[ii * p + kk] * B[kk * n + jj];
                            C[ii * n + jj] += sum;
                        }
                    }
                }

            }
        }
    }
}

int main(){

    float* A = (float*)aligned_alloc( 64,m*p*sizeof(float));
    float* B = (float*)aligned_alloc( 64,p*n*sizeof(float));
    float* C = (float*)aligned_alloc( 64,m*n*sizeof(float));

    memset(A, 1.5, m*p*sizeof(float)); // Lleno la matriz con 1.5
    memset(B, 1.5, p*n*sizeof(float)); // Lleno la matriz con 1.5
    memset(C, 0.0, m*n*sizeof(float)); // Lleno la matriz con 0.0

    double tiempo_transcurrido;

    //_________________________________________________________________

    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_row_col_it(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento row-col-it demoro %fs \n", tiempo_transcurrido);


    //_________________________________________________________________
    
    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_col_row_it(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento col_row_it demoro %fs \n", tiempo_transcurrido);

    //_________________________________________________________________

    memset(C, 1.5, m*n); // Lleno la matriz con 1.5

    
    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_row_it_col(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento row_it_col demoro %fs \n", tiempo_transcurrido);

    //_________________________________________________________________

    memset(C, 1.5, m*n); // Lleno la matriz con 1.5

    
    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_col_it_row(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento col_it_row demoro %fs \n", tiempo_transcurrido);

    //_________________________________________________________________
    
    memset(C, 1.5, m*n); // Lleno la matriz con 1.5


    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_it_row_col(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento it_row_col demoro %fs \n", tiempo_transcurrido);

    //_________________________________________________________________

    memset(C, 1.5, m*n); // Lleno la matriz con 1.5

    
    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_it_col_row(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento it_col_row demoro %fs \n", tiempo_transcurrido);

    //_________________________________________________________________


    
    clock_gettime(CLOCK_MONOTONIC, &comienzoTimer);
        reordenamiento_row_col_it_con_blocking(A,B,C);
    clock_gettime(CLOCK_MONOTONIC, &finTimer);

    
    tiempo_transcurrido = (finTimer.tv_sec - comienzoTimer.tv_sec) +
                          (finTimer.tv_nsec - comienzoTimer.tv_nsec) / 1e9;


    printf("Reordenamiento row_col_it_con_blocking demoro %fs \n", tiempo_transcurrido);

    free(A);
    free(B);
    free(C);
}

