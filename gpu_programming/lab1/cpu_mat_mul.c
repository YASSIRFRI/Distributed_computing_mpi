#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);  // Values between 0 and 99
    }
}

void matrix_multiply(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    float *A, *B, *C;
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory for matrices on CPU
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices A and B with random values
    srand(time(NULL));  // Initialize random seed
    initialize_matrix(A, N * N);
    initialize_matrix(B, N * N);

    // Measure CPU execution time
    start = clock();
    
    // Perform matrix multiplication on CPU
    matrix_multiply(A, B, C, N);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    // Display partial result (to avoid flooding the output)
    printf("Some elements of the resulting matrix C:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.2f ", C[i * N + j]);
        }
        printf("\n");
    }
    
    printf("\nCPU computation time: %f seconds\n", cpu_time_used);

    // Free the allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}