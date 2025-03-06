#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to allocate a 2D matrix
double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

// Function to free a 2D matrix
void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to initialize a matrix with random values
void initialize_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 100; // Random values between 0 and 99
        }
    }
}

// Function to perform block matrix multiplication
void block_matrix_multiply(double** A, double** B, double** C, int n, int B_size) {
    for (int i = 0; i < n; i += B_size) { // Loop over row blocks of C
        for (int j = 0; j < n; j += B_size) { // Loop over column blocks of C
            for (int k = 0; k < n; k += B_size) { // Loop over blocks in A and B
                // Multiply the current block
                for (int ii = i; ii < i + B_size && ii < n; ii++) {
                    for (int jj = j; jj < j + B_size && jj < n; jj++) {
                        for (int kk = k; kk < k + B_size && kk < n; kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 1024; // Size of the matrices (n x n)
    int B_size;   // Block size

    // Allocate and initialize matrices
    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    // Test different block sizes
    printf("Block Size, CPU Time (ms), Memory Bandwidth (MB/s)\n");
    for (B_size = 8; B_size <= 256; B_size *= 2) { // Test block sizes: 8, 16, 32, ..., 256
        // Reset matrix C to 0
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
            }
        }

        // Measure execution time
        clock_t start = clock();
        block_matrix_multiply(A, B, C, n, B_size);
        clock_t end = clock();

        double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

        // Calculate memory bandwidth
        double total_data_accessed = 2.0 * n * n * sizeof(double); // Bytes
        double memory_bandwidth = (total_data_accessed * 1e-6) / (cpu_time / 1000.0); // MB/s

        // Print results
        printf("%d, %f, %f\n", B_size, cpu_time, memory_bandwidth);
    }

    // Free allocated memory
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}