#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void blocked_matrix_multiply(double **A, double **B, double **C, int n, int Bsize) {
    for (int i0 = 0; i0 < n; i0 += Bsize) {
        for (int j0 = 0; j0 < n; j0 += Bsize) {
            for (int k0 = 0; k0 < n; k0 += Bsize) {
                for (int i = i0; i < i0 + Bsize && i < n; i++) {
                    for (int j = j0; j < j0 + Bsize && j < n; j++) {
                        for (int k = k0; k < k0 + Bsize && k < n; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 512;  // Matrix size
    int block_sizes[] = {2,4,8, 16, 32, 64, 128, 256};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    double **A = malloc(n * sizeof(double *));
    double **B = malloc(n * sizeof(double *));
    double **C = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        B[i] = malloc(n * sizeof(double));
        C[i] = calloc(n, sizeof(double));  // zero-initialized
    }

    // Initialize matrices A and B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
        }
    }

    // Print header for CSV output: BlockSize, Time (s), Bandwidth (MB/s)
    printf("BlockSize, Time(s), Bandwidth(MB/s)\n");

    // Loop over the block sizes
    for (int idx = 0; idx < num_block_sizes; idx++) {
        int Bsize = block_sizes[idx];

        // Reset matrix C to zero
        for (int i = 0; i < n; i++) {
            memset(C[i], 0, n * sizeof(double));
        }

        clock_t start = clock();
        blocked_matrix_multiply(A, B, C, n, Bsize);
        clock_t end = clock();
        double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        // Estimate memory bandwidth (for illustration):
        double bandwidth = (3.0 * n * n * n * sizeof(double)) / (cpu_time * 1024 * 1024);

        // Output the results as CSV
        printf("%d, %f, %f\n", Bsize, cpu_time, bandwidth);
    }

    // Free memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
