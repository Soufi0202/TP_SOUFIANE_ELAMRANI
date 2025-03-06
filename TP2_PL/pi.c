#include <stdio.h>
#include <omp.h>

static long num_steps=100000;
double step;

int main() {
    int i,id, nthrds;
    double x,pi;
    double start_time,end_time;
    step = 1.0 / (double) num_steps;
    double sum =0.0;

    start_time = omp_get_wtime();
    
    #pragma omp parallel private(i,x,id) reduction(+:sum)
    {
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        
        for (i = id; i< num_steps; i += nthrds) {
            x = (i + 0.5) *step;
            sum += 4.0 / (1.0+ x *x);
        }
    }
    
    pi = step * sum;
    end_time = omp_get_wtime();
    
    printf("Valeur de pi : %.12f\n", pi);
    printf("Temps CPU : %f sec.\n", end_time-start_time);
    
    return 0;
}
