#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

void matrixVectorMult(double* A, double* b, double* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            x[i] += A[i * size + j] * b[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int size = atoi(argv[1]);
    if (size <= 0) {
        if (rank == 0) {
            printf("Matrix size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int rows_per_proc = size / nprocs;
    int remainder = size % nprocs;
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    
    int row_offset = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    
    double* local_A = (double*)malloc(local_rows * size * sizeof(double));
    double* b = (double*)malloc(size * sizeof(double));
    double* local_x = (double*)malloc(local_rows * sizeof(double));
    double* x_parallel = NULL;
    double* x_serial = NULL;
    double* A = NULL;
    
    if (rank == 0) {
        A = (double*)malloc(size * size * sizeof(double));
        x_parallel = (double*)malloc(size * sizeof(double));
        x_serial = (double*)malloc(size * sizeof(double));
        
        if (!A || !x_parallel || !x_serial) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        srand(42);
        
        int limit = (size < 100) ? size : 100;
        for (int j = 0; j < limit; ++j)
            A[0 * size + j] = (double)rand() / RAND_MAX;
        
        if (size > 1 && size > 100) {
            int copy_len = (size - 100 < 100) ? (size - 100) : 100;
            for (int j = 0; j < copy_len; ++j)
                A[1 * size + (100 + j)] = A[0 * size + j];
        }
        
        for (int i = 0; i < size; ++i)
            A[i * size + i] = (double)rand() / RAND_MAX;
        
        for (int i = 0; i < size; ++i)
            b[i] = (double)rand() / RAND_MAX;
    }
    
    MPI_Bcast(b, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < size; ++j) {
                local_A[i * size + j] = A[i * size + j];
            }
        }
        
        int curr_offset = local_rows;
        for (int p = 1; p < nprocs; ++p) {
            int p_rows = (p < remainder) ? rows_per_proc + 1 : rows_per_proc;
            MPI_Send(&A[curr_offset * size], p_rows * size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            curr_offset += p_rows;
        }
    } else {
        MPI_Status status;
        MPI_Recv(local_A, local_rows * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    double start_time, end_time, parallel_time, serial_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    for (int i = 0; i < local_rows; ++i) {
        local_x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            local_x[i] += local_A[i * size + j] * b[j];
        }
    }
    
    if (rank == 0) {
        for (int i = 0; i < local_rows; ++i) {
            x_parallel[i] = local_x[i];
        }
        
        int curr_offset = local_rows;
        for (int p = 1; p < nprocs; ++p) {
            int p_rows = (p < remainder) ? rows_per_proc + 1 : rows_per_proc;
            MPI_Status status;
            MPI_Recv(&x_parallel[curr_offset], p_rows, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            curr_offset += p_rows;
        }
    } else {
        MPI_Send(local_x, local_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;
    
    if (rank == 0) {
        start_time = MPI_Wtime();
        matrixVectorMult(A, b, x_serial, size);
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;
        
        double max_error = 0.0;
        for (int i = 0; i < size; ++i) {
            double diff = fabs(x_parallel[i] - x_serial[i]);
            if (diff > max_error)
                max_error = diff;
        }
        
        double speedup=serial_time / parallel_time;
        double efficiency=speedup / nprocs;
        
        // Original output format (commented out)
        /*
        printf("CPU time of serial multiplication: %f seconds\n", serial_time);
        printf("CPU time of parallel multiplication using %d processes is %f seconds\n", 
               nprocs, parallel_time);
        printf("Speedup: %f\n", speedup);
        printf("Efficiency: %f\n", efficiency);
        printf("Maximum difference between Parallel and serial result: %e\n", max_error);
        */
        
        // CSV output format (for easy plotting)
        static int first_run = 1;
        if (first_run) {
            printf("Size,Processes,SerialTime,ParallelTime,Speedup,Efficiency,MaxError\n");
            first_run = 0;
        }
        
        printf("%d,%d,%f,%f,%f,%f,%e\n", 
               size, nprocs, serial_time, parallel_time, speedup, efficiency, max_error);
    }
    
    free(local_A);
    free(b);
    free(local_x);
    
    if (rank == 0) {
        free(A);
        free(x_parallel);
        free(x_serial);
    }
    
    MPI_Finalize();
    return 0;
}