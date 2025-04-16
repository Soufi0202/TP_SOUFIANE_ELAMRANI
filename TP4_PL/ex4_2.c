#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int n, m;
    double* A = NULL;
    double* A_local = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 4) {
        if (rank == 0)
            printf("This program requires exactly 4 processes.\n");
        MPI_Finalize();
        return 1;
    }
    
    if (argc == 3) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    } else {
        n = m = 8; // Default size as in the example
    }
    
    if (n % 2 != 0 || m % 2 != 0) {
        if (rank == 0)
            printf("Matrix dimensions must be even for proper splitting.\n");
        MPI_Finalize();
        return 1;
    }
    
    int local_n = n / 2;
    int local_m = m / 2;
    
    A_local = (double*)malloc(local_n * local_m * sizeof(double));
    if (!A_local) {
        printf("Process %d: Memory allocation failed.\n", rank);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        A = (double*)malloc(n * m * sizeof(double));
        if (!A) {
            printf("Process 0: Memory allocation for global matrix failed.\n");
            MPI_Finalize();
            return 1;
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                A[i * m + j] = i * 100 + j; // Value based on position for easy identification
            }
        }
        
        printf("Original %dx%d Matrix A:\n", n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%6.1f ", A[i * m + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int row_start, col_start;
    
    switch (rank) {
        case 0: // Upper-left quadrant
            row_start = 0;
            col_start = 0;
            break;
        case 1: // Upper-right quadrant
            row_start = 0;
            col_start = m / 2;
            break;
        case 2: // Lower-left quadrant
            row_start = n / 2;
            col_start = 0;
            break;
        case 3: // Lower-right quadrant
            row_start = n / 2;
            col_start = m / 2;
            break;
    }
    
    if (rank == 0) {
        // Process 0 extracts its own submatrix
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_m; j++) {
                A_local[i * local_m + j] = A[(i + row_start) * m + (j + col_start)];
            }
        }
        
        for (int p = 1; p < 4; p++) {
            int p_row, p_col;
            switch (p) {
                case 1: // Upper-right
                    p_row = 0;
                    p_col = m / 2;
                    break;
                case 2: // Lower-left
                    p_row = n / 2;
                    p_col = 0;
                    break;
                case 3: // Lower-right
                    p_row = n / 2;
                    p_col = m / 2;
                    break;
            }
            
            // Create a temporary buffer for this process's data
            double* temp = (double*)malloc(local_n * local_m * sizeof(double));
            
            // Extract the appropriate submatrix
            for (int i = 0; i < local_n; i++) {
                for (int j = 0; j < local_m; j++) {
                    temp[i * local_m + j] = A[(i + p_row) * m + (j + p_col)];
                }
            }
            
            // Send the submatrix
            MPI_Send(temp, local_n * local_m, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            
            free(temp);
        }
    } else {
        // Receive submatrix from process 0
        MPI_Status status;
        MPI_Recv(A_local, local_n * local_m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    // Use barrier to ensure all processes have their data
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Each process prints its submatrix
    for (int p = 0; p < 4; p++) {
        if (p == rank) {
            printf("Process %d (i=%d to %d, j=%d to %d):\n", 
                   rank, row_start, row_start + local_n - 1, col_start, col_start + local_m - 1);
            
            for (int i = 0; i < local_n; i++) {
                for (int j = 0; j < local_m; j++) {
                    printf("%6.1f ", A_local[i * local_m + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    free(A_local);
    if (rank == 0) {
        free(A);
    }
    
    MPI_Finalize();
    return 0;
}