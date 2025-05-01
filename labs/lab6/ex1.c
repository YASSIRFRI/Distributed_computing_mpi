#include<mpi.h>
#include<stdlib.h>
#include<stdio.h>


int main(int argc, char** argv){
    int rank, size;
    char grid[4][4];
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int dims[2]= {0,0};
    int periods[2];
    int coords[2];
    int reorder;
    MPI_Comm comm;
    MPI_Dims_create(size, 2, dims);
    periods[0] = 1;
    periods[1] = 1;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);
    if(rank == 0){
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                grid[i][j] = ' ';
            }
        }
        grid[1][1] = 'X';
        grid[1][2] = 'X';
        grid[2][1] = 'X';
        grid[2][2] = 'X';
    }
    MPI_Bcast(grid, 8*8, MPI_CHAR, 0, comm);
    int left, right, up, down;
    MPI_Cart_shift(comm, 0, 1, &left, &right);
    MPI_Cart_shift(comm, 1, 1, &up, &down);
    int left_coords[2], right_coords[2], up_coords[2], down_coords[2];
    MPI_Cart_coords(comm, left, 2, left_coords);
    MPI_Cart_coords(comm, right, 2, right_coords);
    MPI_Cart_coords(comm, up, 2, up_coords);
    MPI_Cart_coords(comm, down, 2, down_coords);
    printf("Process %d at (%d, %d):\n", rank, coords[0], coords[1]);
    printf("Left neighbor: %d at (%d, %d)\n", left, left_coords[0], left_coords[1]);
    printf("Right neighbor: %d at (%d, %d)\n", right, right_coords[0], right_coords[1]);
    printf("Up neighbor: %d at (%d, %d)\n", up, up_coords[0], up_coords[1]);
    printf("Down neighbor: %d at (%d, %d)\n", down, down_coords[0], down_coords[1]);
    int new_grid[4][4];
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            new_grid[i][j] = grid[i][j];
        }
    }
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            int count = 0;
            if(grid[i][j] == 'X'){
                new_grid[i][j] = ' ';
            }else{
                new_grid[i][j] = 'X';
            }
            if(grid[i][j] == 'X'){
                count++;
            }
            if(i > 0 && grid[i-1][j] == 'X'){
                count++;
            }
            if(i < 3 && grid[i+1][j] == 'X'){
                count++;
            }
            if(j > 0 && grid[i][j-1] == 'X'){
                count++;
            }
            if(j < 3 && grid[i][j+1] == 'X'){
                count++;
            }
            if(count < 2 || count > 3){
                new_grid[i][j] = ' ';
            }else if(count == 3){
                new_grid[i][j] = 'X';
            }
        }
    }
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            grid[i][j] = new_grid[i][j];
        }
    }
    printf("Process %d at (%d, %d):\n", rank, coords[0], coords[1]);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            printf("%c ", grid[i][j]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}