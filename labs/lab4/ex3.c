#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size == 1) {
        if (rank == 0) {
            printf("Enter an integer value: ");
            fflush(stdout);
            scanf("%d", &value);
            value += rank;  
            printf("Process %d got %d\n", rank, value);
        }
        MPI_Finalize();
        return 0;
    }
    if (rank == 0) {
        printf("Enter an integer value: ");
        fflush(stdout);
        scanf("%d", &value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += rank;
        printf("Process %d got %d\n", rank, value);
    } else {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += rank;
        printf("Process %d got %d\n", rank, value);
        int next = (rank + 1) % size;
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
