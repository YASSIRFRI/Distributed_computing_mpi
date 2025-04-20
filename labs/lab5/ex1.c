#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 5

void exercise1() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int a[ROWS][COLS];
    int at[COLS][ROWS];
    MPI_Datatype coltype, transtype;
    MPI_Status status;

    if (rank == 0) {
        printf("Process 0 - Matrix a:\n");
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                a[i][j] = i * COLS + j + 1;
                printf("%d ", a[i][j]);
            }
            printf("\n");
        }
    }

    
    MPI_Type_vector(ROWS, 1, COLS, MPI_INT, &coltype);
    MPI_Type_commit(&coltype);

    
    MPI_Type_create_hvector(COLS, 1, ROWS * sizeof(int), coltype, &transtype);
    MPI_Type_commit(&transtype);

    if (rank == 1) {
        MPI_Recv(&at, 1, transtype, 0, 0, MPI_COMM_WORLD, &status);
        printf("\nProcess 1 - Matrix transposee at:\n");
        for (int i = 0; i < COLS; ++i) {
            for (int j = 0; j < ROWS; ++j) {
                printf("%d ", at[i][j]);
            }
            printf("\n");
        }
    } else if (rank == 0) {
        MPI_Send(&a, ROWS * COLS, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&coltype);
    MPI_Type_free(&transtype);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    exercise1();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

