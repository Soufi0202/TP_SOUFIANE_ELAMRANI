#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define N_FEATURES 5
#define LEARNING_RATE 0.01
#define MAX_EPOCHS 1000
#define CONVERGENCE_THRESHOLD 1.0e-2
#define PRINT_INTERVAL 10


typedef struct {
    double x[N_FEATURES];
    double y;
} Sample;


void generate_data(Sample* dataset,int n_samples);
double compute_loss(Sample* local_data,int local_n_samples, double* weights);
void compute_gradient(Sample* local_data,int local_n_samples, double* weights,double* gradient);
void update_weights(double* weights, double* gradient, double learning_rate);

int main(int argc, char** argv) {
    int rank, size,n_samples = 10000; 
    int local_n_samples;
    int *counts = NULL, *displs = NULL;
    Sample *dataset = NULL, *local_dataset = NULL;
    double *weights, *local_gradient, *global_gradient;
    double local_loss,global_loss;
    double start_time,end_time;
    
    // Here I initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) {
        n_samples = atoi(argv[1]);
    }
    
    // Here I m calculating the number of samples per process
    counts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    
    int base_count = n_samples / size;
    int remainder = n_samples % size;
    
    displs[0] = 0;
    for (int i = 0; i < size;i++) {
        counts[i] = base_count+ (i < remainder ? 1 : 0);
        if (i > 0) {
            displs[i] = displs[i-1]+ ounts[i-1];
        }
    }
    
    local_n_samples = counts[rank];
    
    // This is to allocate memory for datasets and the model parameters
    local_dataset = (Sample*)malloc(local_n_samples * sizeof(Sample));
    weights = (double*)calloc(N_FEATURES,sizeof(double));  
    local_gradient = (double*)malloc(N_FEATURES * sizeof(double));
    global_gradient = (double*)malloc(N_FEATURES * sizeof(double));
    

    MPI_Datatype sample_type;
    int blocklengths[2] = {N_FEATURES,1};
    MPI_Aint offsets[2];
    MPI_Datatype types[2] = {MPI_DOUBLE,MPI_DOUBLE};
    
    offsets[0] = offsetof(Sample,x);
    offsets[1] = offsetof(Sample,y);
    

    MPI_Type_create_struct(2, blocklengths, offsets, types,&sample_type);
    MPI_Type_commit(&sample_type);
    
    // I generated dataset on process 0
    if (rank == 0) {
        dataset = (Sample*)malloc(n_samples*sizeof(Sample));
        generate_data(dataset,n_samples);
        printf("Generated %d samples with %d features.\n", n_samples, N_FEATURES);
    }
    
    // Here I started timing
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    

    MPI_Scatterv(dataset, counts, displs, sample_type,local_dataset, local_n_samples, sample_type,0, MPI_COMM_WORLD);
    

    int epoch;
    for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        // First, I compute the local loss and gradient
        local_loss = compute_loss(local_dataset,local_n_samples, weights);
        compute_gradient(local_dataset, local_n_samples, weights,local_gradient);
        
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(local_gradient, global_gradient, N_FEATURES, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        
        // This for normalization
        global_loss /= n_samples;
        for (int i = 0; i < N_FEATURES;i++) {
            global_gradient[i] /= n_samples;
        }
        
        // I update the weights on all processes
        update_weights(weights,global_gradient,LEARNING_RATE);
        
        // Then I print the progress every PRINT_INTERVAL epochs
        if (rank == 0 && (epoch % PRINT_INTERVAL == 0 || epoch == MAX_EPOCHS - 1)) {
            printf("Epoch %d | Loss (MSE): %f | w[0]: %.4f, w[1]: %.4f\n",epoch,global_loss, weights[0], weights[1]);
        }
        
        // Finaly ,I check for convergence
        if (global_loss < CONVERGENCE_THRESHOLD) {
            if (rank == 0) {
                printf("Early stopping at epoch %d - loss %f < %.1e\n", epoch,global_loss, CONVERGENCE_THRESHOLD);
            }
            break;
        }
    }
    
    // Then I end the timing
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Training time: %.3f seconds (MPI)\n",end_time - start_time);
    }
    
    // This section is for the cleaning
    free(local_dataset);
    free(weights);
    free(local_gradient);
    free(global_gradient);
    
    if (rank == 0) {
        free(dataset);
    }
    free(counts);
    free(displs);
    
    MPI_Type_free(&sample_type);
    MPI_Finalize();
    return 0;
}

// method for generating synthetic dataset, following : y = 2*x₁ - x₂ + noise
void generate_data(Sample* dataset, int n_samples) {
    srand(42);
    for (int i =0; i < n_samples; i++) {
        // I generated random feature values between -1 and 1
        for (int j = 0; j < N_FEATURES;j++) {
            dataset[i].x[j] = 2.0* ((double)rand() / RAND_MAX) - 1.0;
        }
        
        // My target value is based on the first two features plus noise
        dataset[i].y = 2.0 * dataset[i].x[0] - 1.0 * dataset[i].x[1];
        
        // The noise that I add is gaussian
        double noise = 0.1 * ((double)rand()/ RAND_MAX - 0.5);
        dataset[i].y += noise;
    }
}

// I implemented this function to compute the Mean Squared Error loss
double compute_loss(Sample* local_data,int local_n_samples, double* weights) {
    double loss = 0.0;
    for (int i = 0; i < local_n_samples; i++) {
        double prediction=0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            prediction += weights[j]*local_data[i].x[j];
        }
        
        double error = prediction-local_data[i].y;
        loss += error * error; 
    }
    return loss;
}

// And this function to compute the gradient of the loss function
void compute_gradient(Sample* local_data, int local_n_samples, double* weights, double* gradient) {
    // First,I initialize the gradient to 0
    for (int j = 0; j < N_FEATURES;j++) {
        gradient[j]=0.0;
    }
    
    // The I compute the gradient components
    for (int i = 0; i < local_n_samples; i++) {
        double prediction = 0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            prediction += weights[j]*local_data[i].x[j];
        }
        
        double error = prediction-local_data[i].y;
        for (int j = 0; j < N_FEATURES; j++) {
            gradient[j] += 2.0 * error *local_data[i].x[j];
        }
    }
}

// This function to update the weights using gradient descent
void update_weights(double* weights, double* gradient, double learning_rate) {
    for (int j =0; j< N_FEATURES; j++) {
        weights[j] -= learning_rate* gradient[j];
    }
}