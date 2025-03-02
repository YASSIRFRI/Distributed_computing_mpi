#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    int i;
    double x, pi, sum = 0.0;
    double start_time, end_time;
    
    step = 1.0/(double) num_steps;
    start_time = omp_get_wtime(); 
    
    #pragma omp parallel for reduction(+:sum) private(x)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0/(1.0 + x*x);
    }
    
    pi = step * sum;
    end_time = omp_get_wtime();  
    
    printf("Pi = %.16f\n", pi);
    printf("Time taken: %f seconds\n", end_time - start_time);
    
    return 0;
}