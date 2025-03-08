#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Utility function declarations
void initializeMatrix(float *matrix, int rows, int cols);
void printMatrixSubset(float *matrix, int rows, int cols, const char *name);
void cpuMatrixMultiply(float *A, float *B, float *C, int m, int n, int k);
void compareResults(float *cpuResult, float *gpuResult, int size);

int main() {
    // Matrix dimensions
    int N = 1024;
    
    // Allocate host memory
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(N * N * sizeof(float));
    float *h_C_gpu = (float*)malloc(N * N * sizeof(float));
    
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initializeMatrix(h_A, N, N);
    initializeMatrix(h_B, N, N);
    
    // Print small subsets to verify data
    printMatrixSubset(h_A, N, N, "Matrix A (subset)");
    printMatrixSubset(h_B, N, N, "Matrix B (subset)");
    
    // ---------------- CPU Matrix Multiplication ----------------
    printf("Performing CPU matrix multiplication...\n");
    
    clock_t cpu_start = clock();
    cpuMatrixMultiply(h_A, h_B, h_C_cpu, N, N, N);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU matrix multiplication completed in %.3f seconds\n", cpu_time);
    
    // ---------------- cuBLAS Matrix Multiplication ----------------
    printf("Performing GPU matrix multiplication with cuBLAS...\n");
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication using cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudaEventRecord(start);
    
    // Note: cuBLAS uses column-major order, so we compute B * A instead of A * B
    // C = alpha*op(A)*op(B) + beta*C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_B, N,  // Matrix B
                d_A, N,  // Matrix A
                &beta, 
                d_C, N); // Matrix C result
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copy result from device to host
    cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate elapsed time
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU matrix multiplication completed in %.3f seconds\n", gpu_time/1000.0);
    
    // Compare results
    printMatrixSubset(h_C_cpu, N, N, "CPU Result (subset)");
    printMatrixSubset(h_C_gpu, N, N, "GPU Result (subset)");
    compareResults(h_C_cpu, h_C_gpu, N * N);
    
    // Print performance comparison
    printf("\nPerformance Comparison:\n");
    printf("CPU time: %.3f seconds\n", cpu_time);
    printf("GPU time: %.3f seconds\n", gpu_time/1000.0);
    printf("Speedup: %.2fx\n", cpu_time/(gpu_time/1000.0));
    
    // Calculate GFLOPS (Giga Floating Point Operations Per Second)
    double cpu_gflops = (2.0 * N * N * N) / (cpu_time * 1e9);
    double gpu_gflops = (2.0 * N * N * N) / ((gpu_time/1000.0) * 1e9);
    printf("CPU Performance: %.2f GFLOPS\n", cpu_gflops);
    printf("GPU Performance: %.2f GFLOPS\n", gpu_gflops);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n--- Complexity Analysis ---\n");
    printf("Native CPU Implementation: O(N³) complexity\n");
    printf("- Triple-nested loops iterating through all matrix elements\n");
    printf("- No optimization for cache locality or parallelism\n\n");
    
    printf("cuBLAS Implementation: Effectively O(N³) but highly optimized\n");
    printf("- Uses tiling to maximize cache utilization\n");
    printf("- Employs thousands of parallel threads on GPU\n");
    printf("- Utilizes specialized matrix multiplication hardware (Tensor Cores if available)\n");
    printf("- Implements advanced blocking strategies to minimize memory access latency\n");
    printf("- Benefits from decades of research in optimizing matrix operations\n");
    
    return 0;
}

// Initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)(rand() % 100) / 100.0f;
        }
    }
}

// Print a small subset of the matrix to verify data
void printMatrixSubset(float *matrix, int rows, int cols, const char *name) {
    printf("%s (3x3 corner):\n", name);
    int display_size = 3;
    for (int i = 0; i < display_size && i < rows; i++) {
        for (int j = 0; j < display_size && j < cols; j++) {
            printf("%.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Native CPU matrix multiplication implementation
void cpuMatrixMultiply(float *A, float *B, float *C, int m, int n, int k) {
    // A: m x k matrix
    // B: k x n matrix
    // C: m x n matrix (result)
    
    // Classic triple loop matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compare CPU and GPU results for accuracy
void compareResults(float *cpuResult, float *gpuResult, int size) {
    float epsilon = 1e-3;  // Tolerance for floating point comparison
    int errors = 0;
    
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > epsilon) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: CPU = %f, GPU = %f\n", 
                      i, cpuResult[i], gpuResult[i]);
            }
        }
    }
    
    if (errors > 0) {
        printf("Found %d errors (tolerance: %e)\n", errors, epsilon);
    } else {
        printf("Results match! No errors found.\n");
    }
}
