#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 50000 

int main() {
    int **matrix;
    long long sum = 0;
    double start_time, end_time, seq_time, par_time;
    int i, j;
    
    matrix = (int**)malloc(SIZE * sizeof(int*));
    for(i = 0; i < SIZE; i++) {
        matrix[i] = (int*)malloc(SIZE * sizeof(int));
    }
    
    srand(time(NULL));
    
    printf("Running sequential version...\n");
    start_time = omp_get_wtime();
    
    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
    
    printf("Matrix preview (sequential):\n");
    for(i = 0; i < 3 && i < SIZE; i++) {
        for(j = 0; j < 3 && j < SIZE; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("...\n");
    }
    printf("...\n");
    
    sum = 0;
    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j++) {
            sum += matrix[i][j];
        }
    }
    
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;
    printf("Sequential sum: %lld\n", sum);
    printf("Sequential execution time: %f seconds\n\n", seq_time);
    
    sum = 0;
    
    printf("Running parallel version...\n");
    start_time = omp_get_wtime();
    
    #pragma omp parallel shared(matrix, sum)
    {
        #pragma omp master
        {
            printf("Thread %d (master) initializing matrix...\n", omp_get_thread_num());
            for(i = 0; i < SIZE; i++) {
                for(j = 0; j < SIZE; j++) {
                    matrix[i][j] = rand() % 100;
                }
            }
        }
        
        #pragma omp barrier
        
        #pragma omp single
        {
            printf("Thread %d (single) printing matrix preview:\n", omp_get_thread_num());
            for(i = 0; i < 3 && i < SIZE; i++) {
                for(j = 0; j < 3 && j < SIZE; j++) {
                    printf("%d\t", matrix[i][j]);
                }
                printf("...\n");
            }
            printf("...\n");
        }
        
        // All threads compute sum in parallel with reduction
        long long local_sum = 0;
        #pragma omp for collapse(2) nowait
        for(i = 0; i < SIZE; i++) {
            for(j = 0; j < SIZE; j++) {
                local_sum += matrix[i][j];
            }
        }
        
        // Use critical section to update the global sum
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    
    end_time = omp_get_wtime();
    par_time = end_time - start_time;
    printf("Parallel sum: %lld\n", sum);
    printf("Parallel execution time: %f seconds\n\n", par_time);
    
    // Compare results
    printf("Comparison:\n");
    printf("Speedup: %f\n", seq_time / par_time);
    
    // Free allocated memory
    for(i = 0; i < SIZE; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return 0;
}