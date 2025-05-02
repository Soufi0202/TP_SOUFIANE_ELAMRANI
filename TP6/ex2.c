#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double f_function(double x, double y) {
    return 2.0 * (x*x - x + y*y - y);
}

double exact_solution(double x, double y) {
    return x * y * (x - 1.0) * (y - 1.0);
}

void init_coefs(double *coef, double hx, double hy) {
    coef[0] = 0.5 * pow(hx, 2) * pow(hy, 2) / (pow(hx, 2) + pow(hy, 2));
    coef[1] = 1.0 / pow(hx, 2);
    coef[2] = 1.0 / pow(hy, 2);
}

void init_grid(double **u, double **f, int local_nx, int local_ny, double hx, double hy, 
               int coords_x, int coords_y, int px, int py) {
    int i, j;
    int start_x = coords_x * (local_nx - 2);
    int start_y = coords_y * (local_ny - 2);
    
    for (i = 0; i <= local_nx + 1; i++) {
        for (j = 0; j <= local_ny + 1; j++) {
            double x = (start_x + i - 1) * hx;
            double y = (start_y + j - 1) * hy;
            
            if (i == 0 || i == local_nx + 1 || j == 0 || j == local_ny + 1) {
                if ((start_x + i - 1 == 0) || (start_x + i - 1 == px * (local_nx - 2)) ||
                    (start_y + j - 1 == 0) || (start_y + j - 1 == py * (local_ny - 2))) {
                    u[i][j] = exact_solution(x, y);
                }
            } else {
                u[i][j] = 0.0;
            }
            
            f[i][j] = f_function(x, y);
        }
    }
}

void exchange_ghost_layers(double **u, int local_nx, int local_ny, 
                          int north, int south, int east, int west, MPI_Comm cart_comm) {
    MPI_Status status;
    int i, j;
    
    double *send_north = (double*)malloc(local_nx * sizeof(double));
    double *send_south = (double*)malloc(local_nx * sizeof(double));
    double *recv_north = (double*)malloc(local_nx * sizeof(double));
    double *recv_south = (double*)malloc(local_nx * sizeof(double));
    double *send_east = (double*)malloc(local_ny * sizeof(double));
    double *send_west = (double*)malloc(local_ny * sizeof(double));
    double *recv_east = (double*)malloc(local_ny * sizeof(double));
    double *recv_west = (double*)malloc(local_ny * sizeof(double));
    
    for (i = 1; i <= local_nx; i++) {
        send_north[i-1] = u[i][1];
        send_south[i-1] = u[i][local_ny];
    }
    
    for (j = 1; j <= local_ny; j++) {
        send_west[j-1] = u[1][j];
        send_east[j-1] = u[local_nx][j];
    }
    
    MPI_Sendrecv(send_north, local_nx, MPI_DOUBLE, north, 0,
                 recv_south, local_nx, MPI_DOUBLE, south, 0,
                 cart_comm, &status);
                 
    MPI_Sendrecv(send_south, local_nx, MPI_DOUBLE, south, 1,
                 recv_north, local_nx, MPI_DOUBLE, north, 1,
                 cart_comm, &status);
    
    MPI_Sendrecv(send_east, local_ny, MPI_DOUBLE, east, 2,
                 recv_west, local_ny, MPI_DOUBLE, west, 2,
                 cart_comm, &status);
                 
    MPI_Sendrecv(send_west, local_ny, MPI_DOUBLE, west, 3,
                 recv_east, local_ny, MPI_DOUBLE, east, 3,
                 cart_comm, &status);
    
    for (i = 1; i <= local_nx; i++) {
        u[i][0] = recv_north[i-1];
        u[i][local_ny+1] = recv_south[i-1];
    }
    
    for (j = 1; j <= local_ny; j++) {
        u[0][j] = recv_west[j-1];
        u[local_nx+1][j] = recv_east[j-1];
    }
    
    free(send_north);
    free(send_south);
    free(recv_north);
    free(recv_south);
    free(send_east);
    free(send_west);
    free(recv_east);
    free(recv_west);
}

double jacobi_iteration(double **u, double **u_new, double **f, double *coef, 
                      int local_nx, int local_ny) {
    int i, j;
    double local_error = 0.0;
    
    for (i = 1; i <= local_nx; i++) {
        for (j = 1; j <= local_ny; j++) {
            u_new[i][j] = coef[0] * (coef[1] * (u[i+1][j] + u[i-1][j]) + 
                                    coef[2] * (u[i][j+1] + u[i][j-1]) - 
                                    f[i][j]);
            
            local_error += pow(u_new[i][j] - u[i][j], 2);
        }
    }
    
    return local_error;
}

void copy_grid(double **u, double **u_new, int local_nx, int local_ny) {
    int i, j;
    
    for (i = 1; i <= local_nx; i++) {
        for (j = 1; j <= local_ny; j++) {
            u[i][j] = u_new[i][j];
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, i, j, iter;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];
    int north, south, east, west;
    MPI_Comm cart_comm;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int global_nx = 12;
    int global_ny = 10;
    int max_iter = 10000;
    double tolerance = 1.0e-6;
    
    MPI_Dims_create(size, 2, dims);
    int px = dims[0];
    int py = dims[1];
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &north, &south);
    
    int local_nx = global_nx / px;
    int local_ny = global_ny / py;
    
    if (coords[0] < global_nx % px) local_nx++;
    if (coords[1] < global_ny % py) local_ny++;
    
    double hx = 1.0 / (global_nx - 1);
    double hy = 1.0 / (global_ny - 1);
    
    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", global_nx, global_ny);
        printf("Topology dimensions: %d along x, %d along y\n", px, py);
        printf("-----------------------------------------\n");
    }
    
    printf("Rank in the topology: %d   Array indices: x from 1 to %d, y from 1 to %d\n", 
           rank, local_nx, local_ny);
    printf("Process %d has neighbors: N %d E %d S %d W %d\n", 
           rank, north, east, south, west);
    
    double **u = (double**)malloc((local_nx + 2) * sizeof(double*));
    double **u_new = (double**)malloc((local_nx + 2) * sizeof(double*));
    double **f = (double**)malloc((local_nx + 2) * sizeof(double*));
    
    for (i = 0; i < local_nx + 2; i++) {
        u[i] = (double*)calloc(local_ny + 2, sizeof(double));
        u_new[i] = (double*)calloc(local_ny + 2, sizeof(double));
        f[i] = (double*)calloc(local_ny + 2, sizeof(double));
    }
    
    double coef[3];
    init_coefs(coef, hx, hy);
    
    init_grid(u, f, local_nx, local_ny, hx, hy, coords[0], coords[1], px, py);
    
    double global_error = 1.0;
    double local_error = 0.0;
    double start_time = MPI_Wtime();
    
    for (iter = 0; iter < max_iter && global_error > tolerance; iter++) {
        exchange_ghost_layers(u, local_nx, local_ny, north, south, east, west, cart_comm);
        
        local_error = jacobi_iteration(u, u_new, f, coef, local_nx, local_ny);
        
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        global_error = sqrt(global_error);
        
        copy_grid(u, u_new, local_nx, local_ny);
        
        if (rank == 0 && (iter % 100 == 0)) {
            printf("Iteration %d\tglobal_error = %.5e\n", iter, global_error);
        }
    }
    
    double elapsed_time = MPI_Wtime() - start_time;
    
    if (rank == 0) {
        printf("Converged after %d iterations in %f seconds\n", iter, elapsed_time);
        printf("Exact solution u_exact - Computed solution u\n");
    }
    
    double local_exact_error = 0.0;
    int start_x = coords[0] * (local_nx - 2);
    int start_y = coords[1] * (local_ny - 2);
    
    for (i = 1; i <= local_nx; i++) {
        for (j = 1; j <= local_ny; j++) {
            double x = (start_x + i - 1) * hx;
            double y = (start_y + j - 1) * hy;
            local_exact_error += pow(u[i][j] - exact_solution(x, y), 2);
        }
    }
    
    double global_exact_error;
    MPI_Reduce(&local_exact_error, &global_exact_error, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    
    if (rank == 0) {
        global_exact_error = sqrt(global_exact_error);
        printf("L2 error between exact and computed solution: %.6e\n", global_exact_error);
        
        double x_samples[5] = {0.1, 0.3, 0.5, 0.7, 0.9};
        for (i = 0; i < 5; i++) {
            printf("%.5e - %.5e\n", exact_solution(x_samples[i], x_samples[i]), 0.0);
        }
    }
    
    for (i = 0; i < local_nx + 2; i++) {
        free(u[i]);
        free(u_new[i]);
        free(f[i]);
    }
    free(u);
    free(u_new);
    free(f);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}