#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100 // Example size (you can change this)

// Function to perform standard matrix multiplication
void matrix_multiply_standard(int n, double **a, double **b, double **c) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void matrix_multiply_optimized(int n, double **a, double **b, double **c) {
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Function to initialize matrices
void initialize_matrices(int n, double **a, double **b, double **c) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 1.0;
            b[i][j] = 2.0;
            c[i][j] = 0.0;
        }
    }
}

// Function to measure execution time
double get_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main() {
    // Allocate memory for matrices
    double **a, **b, **c;
    a = (double **)malloc(N * sizeof(double *));
    b = (double **)malloc(N * sizeof(double *));
    c = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        a[i] = (double *)malloc(N * sizeof(double));
        b[i] = (double *)malloc(N * sizeof(double));
        c[i] = (double *)malloc(N * sizeof(double));
    }

    // Measure execution time for optimized matrix multiplication
    initialize_matrices(N, a, b, c); // Reinitialize matrices
    double start = get_time();
    matrix_multiply_optimized(N, a, b, c);
    double end = get_time();

    double memory_bandwidth = (2.0 * N * N * N * sizeof(double)) / ((end - start) * 1e6); // MB/s

    printf("\nOptimized Matrix Multiplication:\n");
    printf("Execution Time: %.6f seconds\n", end - start);
    printf("Memory Bandwidth: %.2f MB/s\n", memory_bandwidth);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}