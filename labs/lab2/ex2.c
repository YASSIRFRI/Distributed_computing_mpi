// Exercise 2: Parallelizing of PI calculation
// static long num_steps = 100000;
// double step;
// int main ()
// {
// int i; double x, pi, sum = 0.0;
// step = 1.0/(double) num_steps;
// for (i=0;i< num_steps; i++){
// x = (i+0.5)*step;
// sum = sum + 4.0/(1.0+x*x);
// }
// pi = step * sum;
// }
// 1. Create a parallel version of the pi program using a parallel construct.
// 2. Don’t use #pragma parallel for
// 3. Pay close attention to shared versus private variables.
// 4. use double omp_get_wtime() to calculate the CPU time.

#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    int i;
    double pi = 0.0, sum = 0.0;
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    start_time = omp_get_wtime();  
    
    #pragma omp parallel private(i) shared(step, num_steps) reduction(+:sum)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        for (i = thread_id; i < num_steps; i += num_threads) {
            double x = (i + 0.5) * step;
            sum += 4.0/(1.0 + x*x);
        }
    }
    
    pi = step * sum;
    end_time = omp_get_wtime();  
    
    printf("Pi = %.16f\n", pi);
    printf("Time taken: %f seconds\n", end_time - start_time);
    
    return 0;
}
