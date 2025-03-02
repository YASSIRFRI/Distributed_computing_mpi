// In this very simple exercise, you need to :
// 1. Write an OpenMP program displaying the number of threads used for the execution and
// the rank of each of the threads.
// 2. Compile the code manually to create a monoprocessor executable and a parallel executable.
// 3. Test the programs obtained with different numbers of threads for the parallel program,
// without submitting in batch.
// Output example for the parallel program with 4 threads :
// Hello from the rank 2 thread
// Hello from the rank 1 thread
// Hello from the rank 3 thread
// Hello from the rank 0 thread
// Parallel execution of hello_world with 4 threads


#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int thread_rank = omp_get_thread_num();
        printf("Hello from the rank %d thread\n", thread_rank);
        if (thread_rank == 0) {
            printf("Parallel execution of hello_world with %d threads\n", 
                   omp_get_num_threads());
        }
    }
    return 0;
}
