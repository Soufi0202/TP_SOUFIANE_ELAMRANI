#include <stdio.h>
#include <omp.h>

static long num_steps =100000;
double step;

int main() {
    int i;
    double x,pi, sum = 0.0;
    step = 1.0 / (double) num_steps;
    double start_time = omp_get_wtime();
    
    // Parallélisation de la boucle:
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps;i++) {
        x = (i +0.5)*step;
        sum += 4.0 /(1.0 +x* x);
    }
    pi = step*sum;
    double end_time = omp_get_wtime();
    
    printf("Valeur de pi : %.12f\n",pi);
    printf("Temps CPU : %f sec.\n",end_time - start_time);
    return 0;
}
