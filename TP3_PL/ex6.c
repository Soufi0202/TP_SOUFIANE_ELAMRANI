#include <omp.h>
#include <stdio.h>

#define ITER 1000000

int main() {
    int counter = 0;

    // Critical
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < ITER; i++) {
        #pragma omp critical
        counter++;
    }
    printf("Critical Time: %f\n", omp_get_wtime() - start);

    counter = 0;

    // Atomic
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < ITER; i++) {
        #pragma omp atomic
        counter++;
    }
    printf("Atomic Time: %f\n", omp_get_wtime() - start);

    return 0;
}