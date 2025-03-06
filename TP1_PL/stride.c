#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_STRIDE 20

int main() {
    int N = 1000000; // Number of elements in the array
    double *a;

    // Allocate memory for the array
    a = malloc(N * MAX_STRIDE * sizeof(double));
    if (!a) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize the array to 1.0
    for (int i = 0; i < N * MAX_STRIDE; i++) {
        a[i] = 1.0;
    }

    // Print header
    printf("Stride, Sum, Time (msec), Rate (MB/s)\n");

    // Traverse the array with different strides
    for (int i_stride = 1; i_stride <= MAX_STRIDE; i_stride++) {
        double sum = 0.0;
        double start = (double)clock() / CLOCKS_PER_SEC;

        // Perform summation with stride i_stride
        for (int i = 0; i < N * i_stride; i += i_stride) {
            sum += a[i];
        }

        double end = (double)clock() / CLOCKS_PER_SEC;
        double msec = (end - start) * 1000.0; // Time in milliseconds
        double rate = sizeof(double) * N * (1000.0 / msec) / (1024 * 1024); // Memory bandwidth in MB/s

        // Print results
        printf("%d, %f, %f, %f\n", i_stride, sum, msec, rate);
    }

    // Free allocated memory
    free(a);

    return 0;
}