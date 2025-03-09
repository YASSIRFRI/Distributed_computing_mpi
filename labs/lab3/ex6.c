#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 8
#define NUM_INCREMENTS 50000000  

int main() {
    long long counter_critical = 0;
    long long counter_atomic = 0;
    double start_time, end_time;
    double critical_time, atomic_time;
    
    printf("=== Critical vs Atomic for Shared Counters ===\n\n");
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Number of increments per method: %d\n\n", NUM_INCREMENTS);
    omp_set_num_threads(NUM_THREADS);
    printf("Test 1: Using critical section\n");
    printf("----------------------------------------\n");
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long long local_counter = 0;
        #pragma omp for
        for (int i = 0; i < NUM_INCREMENTS; i++) {
            #pragma omp critical
            {
                counter_critical++;
            }
            local_counter++;
            if (local_counter % (NUM_INCREMENTS / (NUM_THREADS * 10)) == 0) {
                printf("Thread %d: Completed %lld increments using critical\n", 
                       thread_id, local_counter);
            }
        }
    }
    end_time = omp_get_wtime();
    critical_time = end_time - start_time;
    printf("Final counter value with critical: %lld\n", counter_critical);
    printf("Time taken with critical section: %.6f seconds\n\n", critical_time);
    printf("Test 2: Using atomic operations\n");
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long long local_counter = 0;
        
        #pragma omp for
        for (int i = 0; i < NUM_INCREMENTS; i++) {
            #pragma omp atomic
            counter_atomic++;
            local_counter++;
            if (local_counter % (NUM_INCREMENTS / (NUM_THREADS * 10)) == 0) {
                printf("Thread %d: Completed %lld increments using atomic\n", 
                       thread_id, local_counter);
            }
        }
    }
    end_time = omp_get_wtime();
    atomic_time = end_time - start_time;
    printf("Final counter value with atomic: %lld\n", counter_atomic);
    printf("Time taken with atomic operations: %.6f seconds\n\n", atomic_time);
    printf("Test 3: Using local counters with reduction (for comparison)\n");
    long long counter_reduction = 0;
    double reduction_time;
    start_time = omp_get_wtime();
    #pragma omp parallel reduction(+:counter_reduction)
    {
        int thread_id = omp_get_thread_num();
        long long local_counter = 0;
        #pragma omp for
        for (int i = 0; i < NUM_INCREMENTS; i++) {
            counter_reduction++;
            local_counter++;
            if (local_counter % (NUM_INCREMENTS / (NUM_THREADS * 10)) == 0) {
                printf("Thread %d: Completed %lld increments using reduction\n", 
                       thread_id, local_counter);
            }
        }
    }
    end_time = omp_get_wtime();
    reduction_time = end_time - start_time;
    printf("Final counter value with reduction: %lld\n", counter_reduction);
    printf("Time taken with reduction: %.6f seconds\n\n", reduction_time);
    printf("=== Performance Comparison ===\n");
    printf("Critical section time:  %.6f seconds\n", critical_time);
    printf("Atomic operation time:  %.6f seconds\n", atomic_time);
    printf("Reduction method time:  %.6f seconds\n", reduction_time);
    
    printf("\nSpeedup of atomic over critical: %.2fx\n", 
           critical_time / atomic_time);
    printf("Speedup of reduction over critical: %.2fx\n", 
           critical_time / reduction_time);
    
    
    
    

    
    return 0;
}