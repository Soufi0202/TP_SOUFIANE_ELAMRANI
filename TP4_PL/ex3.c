#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, value, received_value;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Enter an integer value: ");
        scanf("%d", &value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_value, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, &status);
        
        printf("Process %d: Final value after complete ring traversal is %d\n", rank, received_value);
    } else {
        MPI_Recv(&received_value, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
        value = received_value + rank;
        printf("Process %d: Received %d, added %d, result is %d\n", 
               rank, received_value, rank, value);
        
        int next = (rank + 1) % size;
        
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}