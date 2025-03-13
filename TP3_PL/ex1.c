#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000000

int main() {
    double *A = malloc(N * sizeof(double));
    double sum = 0.0, max = 0.0, std_dev = 0.0;
    int i;

    #pragma omp parallel for
    for (i =0;i< N;i++) {
        A[i] = rand()%100;
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (i= 0; i<N;i++) {
                sum +=A[i];
            }
        }

        #pragma omp section
        {
            max = A[0];
            for (i = 1; i< N;i++) {
                if (A[i] > max) max =A[i];
            }
        }

        #pragma omp section
        {
            double sum_sq = 0.0;
            for (i=0; i< N; i++) {
                sum_sq += A[i]*A[i];
            }
            double mean = sum/ N;
            std_dev = sqrt((sum_sq/ N)-mean *mean);
        }
    }
    printf("Sum: %f, Max: %f, Std Dev: %f\n", sum, max, std_dev);
    free(A);
    return 0;
}