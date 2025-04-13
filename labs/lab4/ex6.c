

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <mpi.h>
 
 int main(int argc, char* argv[])
 {
     MPI_Init(&argc, &argv);
 
     int rank, numprocs;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     if (argc < 2) {
         if (rank == 0) {
             fprintf(stderr, "Usage: %s <N>\n", argv[0]);
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
     long long N = atoll(argv[1]);  
     if (N <= 0) {
         if (rank == 0) {
             fprintf(stderr, "Error: N must be a positive integer.\n");
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
     long long base = N / numprocs;                
     long long remainder = N % numprocs;           
     long long start, end;
     if (rank < remainder) {
         start = rank * (base + 1);
         end   = start + (base + 1);
     } else {
         start = rank * base + remainder;  
         end   = start + base;
     }
     double local_sum = 0.0;
     for (long long i = start; i < end; i++) {
         double x = (i + 0.5) / (double)N;
         local_sum += 1.0 / (1.0 + x * x);
     }
     double global_sum = 0.0;
     MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     if (rank == 0) {
         double pi_approx = (4.0 / (double)N) * global_sum;
         double error = fabs(pi_approx - M_PI);
         printf("Approximation of pi = %.15f\n", pi_approx);
         printf("Actual pi          = %.15f\n", M_PI);
     }
     MPI_Finalize();
     return 0;
 }
 