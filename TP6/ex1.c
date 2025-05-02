#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

void initialize_grid(int **grid, int local_nx, int local_ny, int rank);
void print_grid(int **grid, int local_nx, int local_ny, int rank, int gen);
void update_grid(int **current, int **next, int local_nx, int local_ny);
int count_neighbors(int **grid, int i, int j, int local_nx, int local_ny);
void exchange_ghost_layers(int **grid, int local_nx, int local_ny, int north, int south, int east, int west);

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_nx = 20; 
    int global_ny = 20;
    int generations = 10;

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int px = dims[0];
    int py = dims[1];

    if (rank == 0) {
        printf("Parallel Game of Life with %d processes\n", size);
        printf("Process grid dimensions: %d x %d\n", px, py);
        printf("Global grid dimensions: %d x %d\n", global_nx, global_ny);
        printf("Running for %d generations\n", generations);
    }

    int periods[2] = {1, 1};
    int reorder = 0;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &north, &south);

    if (rank == 0) {
        printf("\nProcess grid established. Beginning simulation...\n\n");
    }

    int local_nx = global_nx / px;
    int local_ny = global_ny / py;

    if (coords[0] < global_nx % px) local_nx++;
    if (coords[1] < global_ny % py) local_ny++;

    int **current_grid = (int**)malloc((local_nx + 2) * sizeof(int*));
    int **next_grid = (int**)malloc((local_nx + 2) * sizeof(int*));
    
    for (int i = 0; i < local_nx + 2; i++) {
        current_grid[i] = (int*)calloc(local_ny + 2, sizeof(int));
        next_grid[i] = (int*)calloc(local_ny + 2, sizeof(int));
    }

    initialize_grid(current_grid, local_nx, local_ny, rank);

    printf("Rank %d at position (%d, %d) with local grid size %d x %d\n", 
           rank, coords[0], coords[1], local_nx, local_ny);
    printf("Rank %d has neighbors: N %d, S %d, E %d, W %d\n", 
           rank, north, south, east, west);

    for (int gen = 0; gen < generations; gen++) {
        exchange_ghost_layers(current_grid, local_nx, local_ny, north, south, east, west);
        update_grid(current_grid, next_grid, local_nx, local_ny);

        if (gen % 5 == 0 || gen == generations - 1) {
            print_grid(current_grid, local_nx, local_ny, rank, gen);
        }

        int **temp = current_grid;
        current_grid = next_grid;
        next_grid = temp;

        MPI_Barrier(cart_comm);
    }

    for (int i = 0; i < local_nx + 2; i++) {
        free(current_grid[i]);
        free(next_grid[i]);
    }
    free(current_grid);
    free(next_grid);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

void initialize_grid(int **grid, int local_nx, int local_ny, int rank) {
    srand(time(NULL) + rank);
    
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            grid[i][j] = (rand() % 100 < 25) ? 1 : 0;
        }
    }
}

void print_grid(int **grid, int local_nx, int local_ny, int rank, int gen) {
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("---------------------------------------------------\n");
        printf("Generation %d\n", gen);
    }
    
    printf("Rank %d - Generation %d:\n", rank, gen);
    for (int i = 1; i <= local_nx; i++) {
        printf("  ");
        for (int j = 1; j <= local_ny; j++) {
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

int count_neighbors(int **grid, int i, int j, int local_nx, int local_ny) {
    int count = 0;
    
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            count += grid[i + di][j + dj];
        }
    }
    
    return count;
}

void update_grid(int **current, int **next, int local_nx, int local_ny) {
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int neighbors = count_neighbors(current, i, j, local_nx, local_ny);
            int state = current[i][j];
            
            if (state == 1 && (neighbors < 2 || neighbors > 3)) {
                next[i][j] = 0;
            } else if (state == 0 && neighbors == 3) {
                next[i][j] = 1;
            } else {
                next[i][j] = state;
            }
        }
    }
}

void exchange_ghost_layers(int **grid, int local_nx, int local_ny, int north, int south, int east, int west) {
    MPI_Status status;
    
    int *send_north = (int*)malloc(local_nx * sizeof(int));
    int *send_south = (int*)malloc(local_nx * sizeof(int));
    int *recv_north = (int*)malloc(local_nx * sizeof(int));
    int *recv_south = (int*)malloc(local_nx * sizeof(int));
    int *send_east = (int*)malloc(local_ny * sizeof(int));
    int *send_west = (int*)malloc(local_ny * sizeof(int));
    int *recv_east = (int*)malloc(local_ny * sizeof(int));
    int *recv_west = (int*)malloc(local_ny * sizeof(int));
    
    for (int i = 1; i <= local_nx; i++) {
        send_north[i-1] = grid[i][1];
        send_south[i-1] = grid[i][local_ny];
    }
    
    for (int j = 1; j <= local_ny; j++) {
        send_west[j-1] = grid[1][j];
        send_east[j-1] = grid[local_nx][j];
    }
    
    MPI_Sendrecv(send_north, local_nx, MPI_INT, north, 0,
                 recv_south, local_nx, MPI_INT, south, 0,
                 MPI_COMM_WORLD, &status);
                 
    MPI_Sendrecv(send_south, local_nx, MPI_INT, south, 1,
                 recv_north, local_nx, MPI_INT, north, 1,
                 MPI_COMM_WORLD, &status);
    
    MPI_Sendrecv(send_east, local_ny, MPI_INT, east, 2,
                 recv_west, local_ny, MPI_INT, west, 2,
                 MPI_COMM_WORLD, &status);
                 
    MPI_Sendrecv(send_west, local_ny, MPI_INT, west, 3,
                 recv_east, local_ny, MPI_INT, east, 3,
                 MPI_COMM_WORLD, &status);
    
    for (int i = 1; i <= local_nx; i++) {
        grid[i][0] = recv_north[i-1];
        grid[i][local_ny+1] = recv_south[i-1];
    }
    
    for (int j = 1; j <= local_ny; j++) {
        grid[0][j] = recv_west[j-1];
        grid[local_nx+1][j] = recv_east[j-1];
    }
    
    MPI_Sendrecv(&grid[1][1], 1, MPI_INT, north+west, 4,
                 &grid[0][0], 1, MPI_INT, south+east, 4,
                 MPI_COMM_WORLD, &status);
                 
    MPI_Sendrecv(&grid[local_nx][1], 1, MPI_INT, north+east, 5,
                 &grid[0][local_ny+1], 1, MPI_INT, south+west, 5,
                 MPI_COMM_WORLD, &status);
                 
    MPI_Sendrecv(&grid[1][local_ny], 1, MPI_INT, south+west, 6,
                 &grid[local_nx+1][0], 1, MPI_INT, north+east, 6,
                 MPI_COMM_WORLD, &status);
                 
    MPI_Sendrecv(&grid[local_nx][local_ny], 1, MPI_INT, south+east, 7,
                 &grid[local_nx+1][local_ny+1], 1, MPI_INT, north+west, 7,
                 MPI_COMM_WORLD, &status);
    
    free(send_north);
    free(send_south);
    free(recv_north);
    free(recv_south);
    free(send_east);
    free(send_west);
    free(recv_east);
    free(recv_west);
}