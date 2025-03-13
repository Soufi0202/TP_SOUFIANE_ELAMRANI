#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 100

int main() {
    int matrix[SIZE][SIZE];
    int sum = 0;

    double start_time =omp_get_wtime();

    #pragma omp master
    for (int i = 0; i<SIZE; i++) {
        for (int j =0; j < SIZE;j++) {
            matrix[i][j] = rand()% 100;
        }
    }

    #pragma omp single
    printf("Matrix initialized.\n");

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE;i++) {
        for (int j = 0;j< SIZE;j++) {
            sum += matrix[i][j];
        }
    }

    double end_time = omp_get_wtime();
    printf("Sum: %d, Time: %f seconds\n", sum, end_time-start_time);
    return 0;
}