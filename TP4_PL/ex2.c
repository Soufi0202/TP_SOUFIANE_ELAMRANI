#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    while (1) {
        if (rank == 0) {
            scanf("%d",&value);
        }
        
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (value < 0) {
            break;
        }
        
        printf("Process %d got %d\n", rank, value);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}

