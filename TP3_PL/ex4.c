#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int data, result;
    int local_sum =0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Enter a number: ");
            scanf("%d", &data);
        }

        #pragma omp barrier

        #pragma omp for reduction(+:local_sum)
        for (int i=0; i< data;i++) {
            local_sum += i;
        }

        #pragma omp barrier

        #pragma omp single
        {
            result=local_sum;
            printf("Result: %d\n", result);
        }
    }
    return 0;
}