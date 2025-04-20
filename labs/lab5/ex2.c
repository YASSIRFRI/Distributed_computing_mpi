#include <stddef.h>
#include <math.h>
#include <time.h>
#include<stdlib.h>
#include<mpi.h>

#define N_FEATURES 5
#define DEFAULT_SAMPLES 1000
#define MAX_EPOCHS 10000
#define LR 0.01
#define THRESHOLD 1e-2

typedef struct {
    double x[N_FEATURES];
    double y;
} Sample;

void generate_data(Sample *data, int n) {
    
    srand(42);
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < N_FEATURES; ++j) {
            data[i].x[j] = rand() / (double)RAND_MAX;
            sum += data[i].x[j];
        }
        data[i].y = sum + ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
    }
}

void exercise2(int argc, char **argv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_samples = (argc > 1) ? atoi(argv[1]) : DEFAULT_SAMPLES;
    Sample *dataset = NULL;
    if (rank == 0) {
        dataset = malloc(num_samples * sizeof(Sample));
        generate_data(dataset, num_samples);
    }

    
    MPI_Datatype sample_type;
    int blocklengths[2] = { N_FEATURES, 1 };
    MPI_Aint displs[2];
    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_DOUBLE };
    Sample temp;
    MPI_Aint base;
    MPI_Get_address(&temp, &base);
    MPI_Get_address(&temp.x, &displs[0]);
    MPI_Get_address(&temp.y, &displs[1]);
    displs[0] -= base;
    displs[1] -= base;
    MPI_Type_create_struct(2, blocklengths, displs, types, &sample_type);
    MPI_Type_commit(&sample_type);

    
    int *counts = malloc(size * sizeof(int));
    int *displacements = malloc(size * sizeof(int));
    int base_count = num_samples / size;
    int rem = num_samples % size;
    for (int i = 0; i < size; ++i) {
        counts[i] = base_count + (i < rem ? 1 : 0);
    }
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + counts[i - 1];
    }

    int local_n = counts[rank];
    Sample *local_data = malloc(local_n * sizeof(Sample));

    
    MPI_Scatterv(dataset, counts, displacements, sample_type,
                 local_data, local_n, sample_type, 0, MPI_COMM_WORLD);

    
    double *w = calloc(N_FEATURES, sizeof(double));
    double *grad = calloc(N_FEATURES, sizeof(double));
    double *global_grad = calloc(N_FEATURES, sizeof(double));
    double local_loss, global_loss;

    double t0 = MPI_Wtime();
    for (int epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
        
        for (int j = 0; j < N_FEATURES; ++j) grad[j] = 0;
        local_loss = 0;
        for (int i = 0; i < local_n; ++i) {
            double pred = 0;
            for (int j = 0; j < N_FEATURES; ++j) pred += w[j] * local_data[i].x[j];
            double err = pred - local_data[i].y;
            local_loss += err * err;
            for (int j = 0; j < N_FEATURES; ++j) grad[j] += 2 * err * local_data[i].x[j];
        }
        
        MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(grad, global_grad, N_FEATURES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_loss /= num_samples;

        
        for (int j = 0; j < N_FEATURES; ++j) {
            w[j] -= LR * global_grad[j] / num_samples;
        }

        
        if (rank == 0) {
            if (epoch % 10 == 0) {
                printf("Epoch %d | Loss (MSE): %.6f | w[0]=%.4f, w[1]=%.4f\n",
                       epoch, global_loss, w[0], w[1]);
            }
            if (global_loss < THRESHOLD) {
                printf("Early stopping at epoch %d — loss %.6f < %.1e\n",
                       epoch, global_loss, THRESHOLD);
                break;
            }
        }
    }
    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("Training time: %.3f seconds (MPI)\n", t1 - t0);
    }

    
    free(local_data);
    free(w);
    free(grad);
    free(global_grad);
    free(counts);
    free(displacements);
    if (rank == 0) free(dataset);
    MPI_Type_free(&sample_type);
}