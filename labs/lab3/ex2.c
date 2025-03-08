#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_VALUES 10
#define MAX_VALUE 100

int main() {
    int values[NUM_VALUES];
    int thread_ids[NUM_VALUES];
    int i, j, temp;
    int temp_id;
    
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for(i = 0; i < NUM_VALUES; i++) {
            values[i] = rand() % MAX_VALUE + 1;
            thread_ids[i] = omp_get_thread_num();
            printf("Thread %d generated value: %d\n", 
                   thread_ids[i], values[i]);
        }


        #pragma omp barrier


        #pragma omp parallel for private(j, temp, temp_id)
        for(i = 0; i < NUM_VALUES; i++) {
            for(j = 0; j < NUM_VALUES - i - 1; j++) {
                if(values[j] > values[j+1]) {
                    temp = values[j];
                    values[j] = values[j+1];
                    values[j+1] = temp;
                    temp_id = thread_ids[j];
                    thread_ids[j] = thread_ids[j+1];
                    thread_ids[j+1] = temp_id;
                }
            }
        } 
        //{
            //printf("\nSorting values...\n\n");
            //for(i = 0; i < NUM_VALUES - 1; i++) {
                //for(j = 0; j < NUM_VALUES - i - 1; j++) {
                    //if(values[j] > values[j+1]) {
                        //temp = values[j];
                        //values[j] = values[j+1];
                        //values[j+1] = temp;
                        //temp_id = thread_ids[j];
                        //thread_ids[j] = thread_ids[j+1];
                        //thread_ids[j+1] = temp_id;
                    //}
                //}
            //}
        //}

        #pragma omp single
        {
            printf("Printing values in ascending order:\n");
            for(i = 0; i < NUM_VALUES; i++) {
                printf("Thread %d generated value: %d\n", 
                       thread_ids[i], values[i]);
            }
        }
    }
    
    return 0;
}