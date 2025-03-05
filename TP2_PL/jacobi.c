#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#ifndef VAL_N
#define VAL_N 120
#endif
#ifndef VAL_D
#define VAL_D 80
#endif


void random_number(double* array,int size) {
    for (int i = 0; i < size; i++)
        array[i] = (double)rand()/(double)(RAND_MAX-1);
}

int main() {
    int n = VAL_N,diag = VAL_D;
    int i,j,iteration = 0;
    double norme;
    

    double *a = (double*)malloc(n * n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *x_courant = (double*)malloc(n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    
    if (!a || !x || !x_courant || !b) {
        fprintf(stderr,"Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    srand(421);
    random_number(a, n*n);
    random_number(b, n);
    
    for (i = 0; i< n; i++) {
        a[i *n +i] += diag;
    }
    for (i = 0;i < n;i++) {
        x[i] = 1.0;
    }
    
    double t_cpu_0 = omp_get_wtime();
    
    while (1) {
        iteration++;
        #pragma omp parallel for private(j) 
        for (i = 0; i<n; i++) {
            double somme = 0.0;
            for (j = 0; j < n; j++) {
                if (j != i) {
                    somme += a[i *n +j] *x[j];
                }
            }
            x_courant[i] = (b[i]-somme) / a[i*n +i];
        }
        double absmax = 0.0;
        #pragma omp parallel for reduction(max:absmax) private(j)
        for (i = 0; i < n; i++) {
            double diff = fabs(x[i]-x_courant[i]);
            if (diff > absmax)
                absmax = diff;
        }
        norme = absmax/n;
        if ((norme <= DBL_EPSILON) || (iteration >= n))
            break;
        memcpy(x,x_courant,n* sizeof(double));
    }
    
    double t_cpu_1 = omp_get_wtime();
    double t_cpu = t_cpu_1-t_cpu_0;
    printf("\n   System size      : %5d\n", n);
    printf("   Iterations       : %4d\n", iteration);
    printf("   Norme            : %10.3E\n", norme);
    printf("   CPU time         : %10.3E sec.\n", t_cpu);
    
    free(a);
    free(x);
    free(x_courant);
    free(b); 
    return EXIT_SUCCESS;
}
