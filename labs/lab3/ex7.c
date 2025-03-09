#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000      
#define CHUNK_SIZE 100000  

void fill_rand(int start, int end, double *A) {
    for (int i = start; i < end; i++) {
        A[i] = rand() % 100;
    }
}

double Sum_array(int start, int end, double *A) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += A[i];
    }
    return sum;
}

int main() {
    double *A, sum = 0.0, runtime;
    int flag = 0;  
    int next_chunk_to_produce = 0;
    int next_chunk_to_consume = 0;
    
    
    A = (double *)malloc(N * sizeof(double));
    if (!A) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    printf("=== Producer-Consumer Problem ===\n\n");
    
    
    printf("Version 1: Sequential approach\n");
    printf("-----------------------------\n");
    
    runtime = omp_get_wtime();
    
    
    fill_rand(0, N, A);
    
    
    sum = Sum_array(0, N, A);
    
    runtime = omp_get_wtime() - runtime;
    printf("Sequential version: In %lf seconds, the sum is %lf\n\n", runtime, sum);
    
    
    printf("Version 2: Basic producer-consumer with flag\n");
    printf("-----------------------------\n");
    
    sum = 0.0;
    flag = 0;
    runtime = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            
            printf("Producer thread started...\n");
            fill_rand(0, N, A);
            
            printf("Producer finished filling array\n");
            
            
            #pragma omp atomic write
            flag = 1;
        }
        
        #pragma omp section
        {
            
            printf("Consumer thread started...\n");
            
            
            int ready = 0;
            while (!ready) {
                #pragma omp atomic read
                ready = flag;
                
                if (!ready) {
                    
                    #pragma omp taskyield
                }
            }
            
            printf("Consumer processing data...\n");
            sum = Sum_array(0, N, A);
            printf("Consumer finished summing array\n");
        }
    }
    
    runtime = omp_get_wtime() - runtime;
    printf("Basic version: In %lf seconds, the sum is %lf\n\n", runtime, sum);
    
    
    printf("Version 3: Chunked producer-consumer\n");
    printf("-----------------------------\n");
    
    sum = 0.0;
    next_chunk_to_produce = 0;
    next_chunk_to_consume = 0;
    int num_chunks = N / CHUNK_SIZE;
    int *chunk_status = (int *)calloc(num_chunks, sizeof(int));  
    
    runtime = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            
            printf("Producer thread started (chunked version)...\n");
            
            for (int chunk = 0; chunk < num_chunks; chunk++) {
                int start = chunk * CHUNK_SIZE;
                int end = (chunk + 1) * CHUNK_SIZE;
                if (end > N) end = N;
                
                
                fill_rand(start, end, A);
                printf("Producer filled chunk %d (%d to %d)\n", chunk, start, end-1);
                
                
                #pragma omp atomic write
                chunk_status[chunk] = 1;
            }
            
            printf("Producer finished all chunks\n");
        }
        
        #pragma omp section
        {
            
            printf("Consumer thread started (chunked version)...\n");
            double local_sum = 0.0;
            
            for (int chunk = 0; chunk < num_chunks; chunk++) {
                int start = chunk * CHUNK_SIZE;
                int end = (chunk + 1) * CHUNK_SIZE;
                if (end > N) end = N;
                
                
                int chunk_ready = 0;
                while (!chunk_ready) {
                    #pragma omp atomic read
                    chunk_ready = chunk_status[chunk];
                    
                    if (!chunk_ready) {
                        
                        #pragma omp taskyield
                    }
                }
                
                
                double chunk_sum = Sum_array(start, end, A);
                local_sum += chunk_sum;
                printf("Consumer processed chunk %d (%d to %d), chunk sum: %lf\n", 
                       chunk, start, end-1, chunk_sum);
            }
            
            sum = local_sum;
            printf("Consumer finished all chunks\n");
        }
    }
    
    runtime = omp_get_wtime() - runtime;
    printf("Chunked version: In %lf seconds, the sum is %lf\n\n", runtime, sum);
    
    
    printf("Version 4: Multiple producers and consumers\n");
    printf("-----------------------------\n");
    
    sum = 0.0;
    for (int i = 0; i < num_chunks; i++) {
        chunk_status[i] = 0;  
    }
    
    runtime = omp_get_wtime();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int is_producer = (thread_id % 2 == 0);  
        
        if (is_producer) {
            
            printf("Thread %d is a producer\n", thread_id);
            
            
            for (int chunk = thread_id / 2; chunk < num_chunks; chunk += num_threads / 2) {
                int start = chunk * CHUNK_SIZE;
                int end = (chunk + 1) * CHUNK_SIZE;
                if (end > N) end = N;
                
                
                fill_rand(start, end, A);
                printf("Producer %d filled chunk %d (%d to %d)\n", 
                       thread_id, chunk, start, end-1);
                
                
                #pragma omp atomic write
                chunk_status[chunk] = 1;
            }
        }
        else {
            
            printf("Thread %d is a consumer\n", thread_id);
            double local_sum = 0.0;
            
            
            for (int chunk = (thread_id - 1) / 2; chunk < num_chunks; chunk += num_threads / 2) {
                int start = chunk * CHUNK_SIZE;
                int end = (chunk + 1) * CHUNK_SIZE;
                if (end > N) end = N;
                
                
                int chunk_ready = 0;
                while (!chunk_ready) {
                    #pragma omp atomic read
                    chunk_ready = chunk_status[chunk];
                    
                    if (!chunk_ready) {
                        
                        #pragma omp taskyield
                    }
                }
                
                
                double chunk_sum = Sum_array(start, end, A);
                local_sum += chunk_sum;
                printf("Consumer %d processed chunk %d (%d to %d), chunk sum: %lf\n", 
                       thread_id, chunk, start, end-1, chunk_sum);
            }
            
            
            #pragma omp atomic
            sum += local_sum;
        }
    }
    
    runtime = omp_get_wtime() - runtime;
    printf("Multi-threaded version: In %lf seconds, the sum is %lf\n", runtime, sum);
    
    
    free(A);
    free(chunk_status);
    
    return 0;
}