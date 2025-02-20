#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_STRIDE 20

int main(void) {
    int N = 1000000;
    double *a = malloc(N * MAX_STRIDE * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    
    for (int i = 0; i < N * MAX_STRIDE; i++) {
        a[i] = 1.0;
    }

    FILE *fprin = fopen("o2.txt", "w");
    if (fprin == NULL) {
        fprintf(stderr, "Error opening file.\n");
        free(a);
        return 1;
    }

    fprintf(fprin, "stride, sum, time (msec), rate (MB/s)\n");

    for (int i_stride = 1; i_stride <= MAX_STRIDE; i_stride++) {
        double sum = 0.0;
        clock_t start = clock();
        
        // Loop using the given stride
        for (int i = 0; i < N * i_stride; i += i_stride) {
            sum += a[i];
        }
        
        clock_t end = clock();
        double msec = ((double)(end - start)) * 1000.0 / CLOCKS_PER_SEC;
        double rate = (sizeof(double) * N * (1000.0 / msec)) / (1024.0 * 1024.0);
        
        fprintf(fprin, "%d, %f, %f, %f\n", i_stride, sum, msec, rate);
    }

    fclose(fprin);
    free(a);
    return 0;
}
