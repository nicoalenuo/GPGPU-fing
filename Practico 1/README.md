# Practico 1, GPU

## Compilacion:
- Utilizar los comandos `make all` y `make clean` para compilar todos los archivos y eliminar los ejecutables respectivamente.
- Usamos la flag -O0 para que el compilador no realize optimizaciones sobre el codigo al compilar.

## Programas incluidos:
- velocidadAccesoCache.cpp (Ejercicio1 A)
  - Calcula la velocidad de acceso a cada nivel de memoria.
  - Es posible cambiar los valores de las constantes CACHE_L1_SIZE y CACHE_L2_SIZE para adecuar el programa al sistema donde se ejecute.
  - La constante SIZE determina el tamaño del array sobre el cual se realizan las lecturas, se debe mantener su tamaño menor a `CACHE_L1_SIZE`.
- multiplicacionMatrices.cpp
  - Multiplica dos matrices (A de mxp y B de pxn, el resultado es almacenado en C de mxn) donde cada entrada de cada una de ellas contiene el valor 1.5.
  - Es posible cambiar los valores de las constantes `BLOCK_SIZEm`, `BLOCK_SIZEn`, `BLOCK_SIZEp`, `m`, `n` y `p` según se requiera.
  - Se realizan 7 multiplicaciones de estas matrices, bajo 6 distintos ordenamientos de bucles y utilizando blocking.
- prefetchHardware.cpp
  - Se registra el tiempo requerido para leer todos las entradas de un arreglo con y sin utilizar prefetch por hardware.
  - La constante  `tam_array` determina el tamaño del array a leer, se espera que este sea un múltiple de `tam_bloque`, que corresponde al tamaño de línea de cache en la cpu utilizada.
- prefetchSoftware.cpp
  - Se recorre una matriz siguiendo un patron no lineal, a la vez que se hace prefetch de los dos siguientes bloques a recorrer .
  - Se compara el tiempo que se tarda al ejecutar el mismo codigo  sin efectuar prefetch.
  - `tam_fila`, determina la cantidad de elementos por fila de la matriz a recorrer, la matriz es cuadrada.
- alineoMemoria.cpp
  - Compara el rendimiento al acceder a arrays cuya única diferencia es su alineación en memoria.
  - Las constantes `ARRAY_SIZE` y `CANT_ITER` determinan la cantidad de elementos (cada elemento de 48bytes) del array y la cantidad de veces que se recorre cada array