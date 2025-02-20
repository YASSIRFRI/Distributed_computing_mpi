
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply_standard(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    int n = 1028; 
    double **a, **b, **c;
    a = malloc(n * sizeof(double*));
    b = malloc(n * sizeof(double*));
    c = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        a[i] = malloc(n * sizeof(double));
        b[i] = malloc(n * sizeof(double));
        c[i] = calloc(n, sizeof(double));
    }

    // Initialize matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 1.0;
            b[i][j] = 1.0;
        }
    }

    clock_t start = clock();
    matrix_multiply_standard(a, b, c, n);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Standard multiplication time: %f seconds\n", cpu_time);

    // Free memory
    for (int i = 0; i < n; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);
    return 0;
}

