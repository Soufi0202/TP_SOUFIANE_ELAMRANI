#include <omp.h>
#include <stdio.h>
#include <windows.h> 

#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE], count = 0;
omp_lock_t lock;

void producer() {
    for (int i = 0; i < 20; i++) {
        omp_set_lock(&lock);
        if (count < BUFFER_SIZE) {
            buffer[count++] = i;
            printf("Produced: %d\n", i);
        }
        omp_unset_lock(&lock);
        usleep(100);
    }
}

void consumer() {
    for (int i = 0; i < 20; i++) {
        omp_set_lock(&lock);
        if (count > 0) {
            int item = buffer[--count];
            printf("Consumed: %d\n", item);
        }
        omp_unset_lock(&lock);
        usleep(100);
    }
}

int main() {
    omp_init_lock(&lock);
    #pragma omp parallel sections
    {
        #pragma omp section { producer(); }
        #pragma omp section { consumer(); }
    }
    omp_destroy_lock(&lock);
    return 0;
}