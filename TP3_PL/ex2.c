#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int count = 0;
    int num_threads = 4;
    int numbers[4];
    int sorted[4];

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        numbers[tid] = rand() %100;

        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 0; i< num_threads;i++) sorted[i] = numbers[i];
            for (int i = 0; i<num_threads-1;i++) {
                for (int j = 0; j < num_threads-i-1; j++) {
                    if (sorted[j] > sorted[j+1]) {
                        int temp = sorted[j];
                        sorted[j] = sorted[j+1];
                        sorted[j+1] = temp;
                    }
                }
            }
        }

        while (count < num_threads) {
            #pragma omp critical
            {
                if (numbers[tid]==sorted[count]) {
                    printf("Thread %d generated value: %d\n", tid,numbers[tid]);
                    count++;
                }
            }
        }
    }
    return 0;
}