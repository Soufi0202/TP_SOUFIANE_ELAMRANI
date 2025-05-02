#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROWS 4
#define COLS 5

void print_matrix(int rows, int cols, int matrix[rows][cols],int rank, const char* name) {
    printf("Process %d - Matrix %s:\n", rank, name);
    for (int i = 0; i<rows;i++) {
        for (int j=0; j <cols; j++) {
            printf("%2d ",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    int rank,size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            printf("My program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        int a[ROWS][COLS];
        int value = 1;
        for (int i = 0; i < ROWS; i++) {
            for (int j=0; j < COLS; j++) {
                a[i][j] = value++;
            }
        }
        
        print_matrix(ROWS, COLS, a, rank, "a");
        
        MPI_Send(&a[0][0], ROWS * COLS, MPI_INT,1,0, MPI_COMM_WORLD);
    } 

    else if (rank == 1) {
        int at[COLS][ROWS];
        MPI_Datatype column_type,transpose_type;
        
        MPI_Type_vector(ROWS, 1,COLS, MPI_INT, &column_type);
        
        MPI_Type_create_hvector(COLS,1, sizeof(int),column_type, &transpose_type);
        
        MPI_Type_commit(&transpose_type);
        
        MPI_Recv(&at[0][0],1,transpose_type,0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Type_free(&column_type);
        MPI_Type_free(&transpose_type);
        print_matrix(COLS,ROWS, at, rank,"transposee at");
    }
    
    MPI_Finalize();
    return 0;
}