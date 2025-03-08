#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define N 10000000

int main() {
    double *array = (double*)malloc(N * sizeof(double));
    double sum = 0.0, max = -INFINITY, stddev = 0.0;
    double mean = 0.0;
    int i;
    
    srand(time(NULL));
    
    printf("Initializing array with random values...\n");
    for(i = 0; i < N; i++) {
        array[i] = (double)rand() / RAND_MAX * 100.0;
    }
    
    for(i = 0; i < N; i++) {
        mean += array[i];
    }
    mean /= N;
    
    double start_time = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double local_sum = 0.0;
            for(i = 0; i < N; i++) {
                local_sum += array[i];
            }
            sum = local_sum;
            printf("Thread %d calculated sum: %f\n", omp_get_thread_num(), sum);
        }
        
        #pragma omp section
        {
            double local_max = array[0];
            for(i = 1; i < N; i++) {
                if(array[i] > local_max) {
                    local_max = array[i];
                }
            }
            max = local_max;
            printf("Thread %d calculated maximum: %f\n", omp_get_thread_num(), max);
        }
        
        #pragma omp section
        {
            double local_variance = 0.0;
            for(i = 0; i < N; i++) {
                local_variance += (array[i] - mean) * (array[i] - mean);
            }
            stddev = sqrt(local_variance / N);
            printf("Thread %d calculated standard deviation: %f\n", omp_get_thread_num(), stddev);
        }
    }
    
    double end_time = omp_get_wtime();
    
    printf("\nResults:\n");
    printf("Sum: %f\n", sum);
    printf("Max: %f\n", max);
    printf("Standard Deviation: %f\n", stddev);
    printf("Execution time: %f seconds\n", end_time - start_time);
    
    free(array);
    return 0;
}