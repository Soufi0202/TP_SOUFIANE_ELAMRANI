#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int tid =omp_get_thread_num();
        int nbThreads = omp_get_num_threads();
        printf("Hello from the rank %d thread\n", tid);
        
        #pragma omp master
        {
            printf("Parallel execution of hello_world with %d threads\n",nbThreads);
        }
    }
    return 0;
}
