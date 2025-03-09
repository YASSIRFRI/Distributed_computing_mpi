#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>  


void task_a(int iterations) {
    double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        result += i * 0.5;
        if (i % 1000000 == 0) usleep(1);  
    }
    printf("Task A result: %.2f\n", result);
}


void task_b(int iterations) {
    double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        result += sin(i * 0.001) * cos(i * 0.0005);
        if (i % 500000 == 0) usleep(1);  
    }
    printf("Task B result: %.2f\n", result);
}


void task_c(int iterations) {
    double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        result += sin(i * 0.001) * cos(i * 0.0005) * tan(i * 0.0001);
        if (i % 200000 == 0) usleep(2);  
    }
    printf("Task C result: %.2f\n", result);
}
int main() {
    double start_time, end_time;
    
    printf("=== Load Balancing with Parallel Sections ===\n\n");
    
    
    
    
    printf("Run 1: Default workload distribution\n");
    printf("----------------------------------------\n");
    
    
    int task_a_load = 5000000;   
    int task_b_load = 20000000;  
    int task_c_load = 40000000;  
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d starting Task A (light computation)...\n", omp_get_thread_num());
            task_a(task_a_load);
            printf("Thread %d completed Task A\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting Task B (moderate computation)...\n", omp_get_thread_num());
            task_b(task_b_load);
            printf("Thread %d completed Task B\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting Task C (heavy computation)...\n", omp_get_thread_num());
            task_c(task_c_load);
            printf("Thread %d completed Task C\n", omp_get_thread_num());
        }
    }
    
    end_time = omp_get_wtime();
    printf("Run 1 completed in %.6f seconds\n\n", end_time - start_time);
    
    
    
    
    printf("Run 2: Split heavy task into multiple sections\n");
    printf("----------------------------------------\n");
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d starting Task A (light)...\n", omp_get_thread_num());
            task_a(task_a_load);
            printf("Thread %d completed Task A\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting Task B (moderate)...\n", omp_get_thread_num());
            task_b(task_b_load);
            printf("Thread %d completed Task B\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting Task C part 1 (heavy split)...\n", omp_get_thread_num());
            task_c(task_c_load / 2);
            printf("Thread %d completed Task C part 1\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting Task C part 2 (heavy split)...\n", omp_get_thread_num());
            task_c(task_c_load / 2);
            printf("Thread %d completed Task C part 2\n", omp_get_thread_num());
        }
    }
    
    end_time = omp_get_wtime();
    printf("Run 2 completed in %.6f seconds\n\n", end_time - start_time);
    
    
    
    
    printf("Run 3: Balanced workload distribution\n");
    printf("----------------------------------------\n");
    
    
    int total_work = task_a_load + task_b_load + task_c_load;
    int balanced_load = total_work / 4;  
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d starting balanced work unit 1...\n", omp_get_thread_num());
            
            task_a(balanced_load / 2);
            task_b(balanced_load / 4);
            task_c(balanced_load / 4);
            printf("Thread %d completed balanced work unit 1\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting balanced work unit 2...\n", omp_get_thread_num());
            task_b(balanced_load / 2);
            task_c(balanced_load / 2);
            printf("Thread %d completed balanced work unit 2\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting balanced work unit 3...\n", omp_get_thread_num());
            task_a(balanced_load / 3);
            task_c(2 * balanced_load / 3);
            printf("Thread %d completed balanced work unit 3\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d starting balanced work unit 4...\n", omp_get_thread_num());
            task_a(balanced_load / 3);
            task_b(balanced_load / 3);
            task_c(balanced_load / 3);
            printf("Thread %d completed balanced work unit 4\n", omp_get_thread_num());
        }
    }
    
    end_time = omp_get_wtime();
    printf("Run 3 completed in %.6f seconds\n\n", end_time - start_time);
    return 0;
}


