 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
  
 void matrixVectorMulSerial(const double* A, const double* x, double* y, int size)
 {
     for (int i = 0; i < size; i++) {
         double sum = 0.0;
         for (int j = 0; j < size; j++) {
             sum += A[i * size + j] * x[j];
         }
         y[i] = sum;
     }
 }
 int main(int argc, char* argv[])
 {
     MPI_Init(&argc, &argv);
     int rank, numprocs;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     if (argc < 2) {
         if (rank == 0) {
             fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
 
     int N = atoi(argv[1]);
     if (N <= 0) {
         if (rank == 0) {
             fprintf(stderr, "Error: matrix_size must be a positive integer.\n");
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
 

     double *A = NULL;          
     double *x = NULL;          
     double *y = NULL;          
     double *y_serial = NULL;   
     double *local_A = NULL;    
     double *local_y = NULL;    
 
     
     
     int rows_per_proc = N / numprocs;
 
     if (rank == 0) {
         
         A = (double *) malloc(N * N * sizeof(double));
         x = (double *) malloc(N * sizeof(double));
         y = (double *) malloc(N * sizeof(double));
         y_serial = (double *) malloc(N * sizeof(double));
 
         if (!A || !x || !y || !y_serial) {
             fprintf(stderr, "Error: memory allocation failed on rank 0.\n");
             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
 
         
         
         srand(42);
         for (int i = 0; i < N; i++) {
             x[i] = (double) rand() / (double) RAND_MAX;  
             for (int j = 0; j < N; j++) {
                 A[i*N + j] = (double) rand() / (double) RAND_MAX;
             }
         }
     }
 
     
     local_A = (double *) malloc(rows_per_proc * N * sizeof(double));
     local_y = (double *) malloc(rows_per_proc * sizeof(double));
     if (!local_A || !local_y) {
         fprintf(stderr, "Error: memory allocation failed on rank %d.\n", rank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
 
     MPI_Scatter(
          A,
          rows_per_proc * N,
          MPI_DOUBLE,
          local_A,
          rows_per_proc * N,
          MPI_DOUBLE,
          0,
         MPI_COMM_WORLD
     );
     if (rank != 0) {
         
         x = (double *) malloc(N * sizeof(double));
         if (!x) {
             fprintf(stderr, "Error: memory allocation for x failed on rank %d.\n", rank);
             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
     }
      MPI_Bcast(
          x,
          N,
          MPI_DOUBLE,
          0,
         MPI_COMM_WORLD
     );
     if (rank == 0) {
         matrixVectorMulSerial(A, x, y_serial, N);
     }
     double start_time = MPI_Wtime();  
     for (int i = 0; i < rows_per_proc; i++) {
         double sum = 0.0;
         for (int j = 0; j < N; j++) {
             sum += local_A[i*N + j] * x[j];
         }
         local_y[i] = sum;
     }
     double end_time = MPI_Wtime();
     double local_elapsed = end_time - start_time;
     double max_time = 0.0;
     MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
     MPI_Gather(
          local_y,
 rows_per_proc,
 MPI_DOUBLE,
          y,
 rows_per_proc,
 MPI_DOUBLE,
          0,
         MPI_COMM_WORLD
     );
     if (rank == 0) {
         double max_diff = 0.0;
         for (int i = 0; i < N; i++) {
             double diff = fabs(y[i] - y_serial[i]);
             if (diff > max_diff) {
                 max_diff = diff;
             }
         }
         printf("CPU time of parallel multiplication using %d processes = %f seconds\n",
                numprocs, max_time);
         printf("Maximum difference between Parallel and Serial result: %e\n", max_diff);
     }
     free(local_A);
     free(local_y);
     free(x);  
     if (rank == 0) {
         free(A);
         free(y);
         free(y_serial);
     }
 
     MPI_Finalize();
     return 0;
 }
 