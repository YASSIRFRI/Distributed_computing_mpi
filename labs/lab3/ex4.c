#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 10000

int main() {
    int n;
    int *data = NULL;
    long long result = 0;
    double start_time, end_time;
    
    printf("Starting multi-stage computation with barrier synchronization\n");
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel shared(data, n, result)
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        
        #pragma omp single
        {
            n = ARRAY_SIZE;
            printf("Stage 1: Thread %d reading input data (n = %d)...\n", thread_id, n);
            
            data = (int*)malloc(n * sizeof(int));
            if (!data) {
                fprintf(stderr, "Memory allocation failed!\n");
                exit(1);
            }
            for (int i = 0; i < n; i++) {
                data[i] = i % 100; 
            }
            printf("Stage 1 complete: Thread %d finished reading data\n", thread_id);
        }
        
        
        long long local_sum = 0;
        int items_per_thread = n / thread_count;
        int start_idx = thread_id * items_per_thread;
        int end_idx = (thread_id == thread_count - 1) ? n : start_idx + items_per_thread;
        
        printf("Stage 2: Thread %d processing data from index %d to %d\n", 
               thread_id, start_idx, end_idx - 1);
        for (int i = start_idx; i < end_idx; i++) {
            local_sum += data[i] * data[i];
        }
        
        #pragma omp critical
        {
            result += local_sum;
        }
        #pragma omp barrier
        
        #pragma omp single
        {
            printf("Stage 3: Thread %d writing final result\n", thread_id);
            printf("Final result: %lld\n", result);
            
            free(data);
            
            end_time = omp_get_wtime();
            printf("Total execution time: %.6f seconds\n", end_time - start_time);
        }
    }
    
    return 0;
}