#include <omp.h>
#include <stdio.h>
#include <windows.h>  

void task_a() { Sleep(1000); }  
void task_b() { Sleep(2000); } 
void task_c() { Sleep(3000); } 

int main() {
    double start =omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        { task_a(); }

        #pragma omp section
        { task_b(); }

        #pragma omp section
        { task_c(); }
    }

    printf("Time: %f\n", omp_get_wtime()-start);
    return 0;
}