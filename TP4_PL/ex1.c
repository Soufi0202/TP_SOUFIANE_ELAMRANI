#include <stdio.h>
#include <mpi.h>
 
int main(int argc, char** argv) {
    int rank,size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
     
    if (rank == 0) {
        printf("Hello World from the master process (rank 0). Total processes: %d\n", size);
    }
     
    MPI_Finalize();
    return 0;
}


/*
 * Part 4: What happens if MPI_Finalize is omitted?
 * 
 * If we omit MPI_Finalize():
 * 1. The program may not terminate properly
 * 2. Resources allocated by MPI won't be released
 * 3. In some MPI implementations, the program might hang or terminate abnormally
 * 4. It's considered improper MPI programming and can cause issues especially in larger applications
 * 5. Some MPI implementations might report errors or warnings
 */