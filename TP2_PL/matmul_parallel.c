#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int m = 512, n = 512;
    int i,j,k;
    
    double *a = (double *) malloc(m * n * sizeof(double));
    double *b = (double *) malloc(n * m * sizeof(double));
    double *c = (double *) malloc(m * m * sizeof(double));

    for (i = 0; i <m; i++)
        for (j = 0;j < n;j++)
            a[i* n+j] = (i +1)+(j + 1);
    
    for (i = 0;i < n; i++)
        for (j = 0;j <m;j++)
            b[i * m+ j] =(i +1) - (j+ 1);
    
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            c[i *m +j] = 0.0;
    
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) private(k) schedule(static,16)
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            for (k = 0; k < n; k++) {
                c[i * m +j] +=a[i* n + k] *b[k * m +j];
            }
        }
    }
    
    double end = omp_get_wtime();
    printf("Temps d'exécution : %f sec.\n", end - start);
    free(a); free(b); free(c);
    return 0;
}
