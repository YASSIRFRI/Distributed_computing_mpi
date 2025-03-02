#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int m = 1000; 
    int n = 1000;
    double start_time, end_time;
    
    double *a = (double *)malloc(m * n * sizeof(double));
    double *b = (double *)malloc(n * m * sizeof(double));
    double *c = (double *)malloc(m * m * sizeof(double));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1); // Access via 1D indexing
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
        }
    }
    
    // Matrix multiplication with OpenMP
    start_time = omp_get_wtime();
    
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    
    end_time = omp_get_wtime();
    
    // Print execution time
    printf("Matrix multiplication time: %f seconds\n", end_time - start_time);
    printf("Number of threads used: %d\n", omp_get_max_threads());
    
    // Verify result by checking a few elements (optional)
    printf("Sample output - c[0][0] = %f\n", c[0]);
    
    // Free allocated memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}