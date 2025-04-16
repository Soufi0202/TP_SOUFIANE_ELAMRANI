#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double calculate_pi_serial(long long n) {
    double sum = 0.0;
    double x;
    
    for (long long i = 0; i < n; i++) {
        x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }
    
    return sum / n;
}

int main(int argc, char* argv[]) {
    int rank, size;
    long long n;
    double pi, pi_serial, local_sum, x;
    double start_time, end_time, parallel_time, serial_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) {
        n = atoll(argv[1]);
    } else {
        n = 1000000000; // Default value
    }
    
    if (rank == 0) {
        printf("Calculating Pi with %lld intervals\n", n);
    }
    
    long long intervals_per_proc = n / size;
    long long remainder = n % size;
    long long my_start, my_end;
    
    // Distribute remaining intervals
    if (rank < remainder) {
        intervals_per_proc++;
        my_start = rank * intervals_per_proc;
    } else {
        my_start = rank * (intervals_per_proc) + remainder;
    }
    my_end = my_start + intervals_per_proc;
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Calculate local sum
    local_sum = 0.0;
    for (long long i = my_start; i < my_end; i++) {
        x = (i + 0.5) / n;
        local_sum += 4.0 / (1.0 + x * x);
    }
    
    // Reduce all local sums to get total sum
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Calculate final value of Pi
    if (rank == 0) {
        pi = global_sum / n;
    }
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;
    
    // Serial calculation for comparison (only on rank 0)
    if (rank == 0) {
        start_time = MPI_Wtime();
        pi_serial = calculate_pi_serial(n);
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;
        
        double speedup = serial_time / parallel_time;
        double efficiency = speedup / size;
        
        printf("Pi approximation (parallel): %.16f\n", pi);
        printf("Pi approximation (serial): %.16f\n", pi_serial);
        printf("Error between parallel and serial: %.16e\n", pi - pi_serial);
        printf("Serial execution time: %.6f seconds\n", serial_time);
        printf("Parallel execution time with %d processes: %.6f seconds\n", size, parallel_time);
        printf("Speedup: %.6f\n", speedup);
        printf("Efficiency: %.6f\n", efficiency);
        
    }
    
    MPI_Finalize();
    return 0;
}