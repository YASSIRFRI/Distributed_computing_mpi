#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(i,j,ny) ((i)*(ny)+(j))
int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int Nx=64,Ny=64;
    double tol=1e-6;
    int max_iter=10000;
    if(argc>=3){
        Nx=atoi(argv[1]);
        Ny=atoi(argv[2]);
    }
    if(argc>=4)max_iter=atoi(argv[3]);
    if(argc>=5)tol=atof(argv[4]);
    int dims[2]={0,0};
    MPI_Dims_create(size,2,dims);
    int periods[2]={0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&cart);
    int coords[2];
    MPI_Cart_coords(cart,rank,2,coords);
    if(Nx%dims[0]||Ny%dims[1]){
        if(rank==0)fprintf(stderr,"grid not divisible\n");
        MPI_Finalize();
        return 0;
    }
    int lx=Nx/dims[0];
    int ly=Ny/dims[1];
    int nrows=lx+2,ncols=ly+2;
    double hx=1.0/(Nx-1);
    double hy=1.0/(Ny-1);
    double coef1=0.5*hx*hx*hy*hy/(hx*hx+hy*hy);
    double coef2=1.0/(hx*hx);
    double coef3=1.0/(hy*hy);
    double *u=calloc(nrows*ncols,sizeof(double));
    double *unew=calloc(nrows*ncols,sizeof(double));
    double *f=calloc(nrows*ncols,sizeof(double));
    for(int i=1;i<=lx;i++){
        for(int j=1;j<=ly;j++){
            int gi=coords[0]*lx+i-1;
            int gj=coords[1]*ly+j-1;
            double x=gi*hx;
            double y=gj*hy;
            f[IDX(i,j,ncols)]=2.0*(x*x-x+y*y-y);
        }
    }
    int north,south,west,east;
    MPI_Cart_shift(cart,0,1,&north,&south);
    MPI_Cart_shift(cart,1,1,&west,&east);
    MPI_Datatype coltype;
    MPI_Type_vector(lx,1,ncols,MPI_DOUBLE,&coltype);
    MPI_Type_commit(&coltype);
    int iter=0;
    double global_err=1.0;
    while(iter<max_iter&&global_err>tol){
        MPI_Sendrecv(&u[IDX(1,1,ncols)],ly,MPI_DOUBLE,north,0,&u[IDX(lx+1,1,ncols)],ly,MPI_DOUBLE,south,0,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[IDX(lx,1,ncols)],ly,MPI_DOUBLE,south,1,&u[IDX(0,1,ncols)],ly,MPI_DOUBLE,north,1,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[IDX(1,1,ncols)],1,coltype,west,2,&u[IDX(1,ly+1,ncols)],1,coltype,east,2,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[IDX(1,ly,ncols)],1,coltype,east,3,&u[IDX(1,0,ncols)],1,coltype,west,3,cart,MPI_STATUS_IGNORE);
        double local_err=0.0;
        for(int i=1;i<=lx;i++){
            for(int j=1;j<=ly;j++){
                unew[IDX(i,j,ncols)]=coef1*(coef2*(u[IDX(i+1,j,ncols)]+u[IDX(i-1,j,ncols)])+coef3*(u[IDX(i,j+1,ncols)]+u[IDX(i,j-1,ncols)])-f[IDX(i,j,ncols)]);
                double diff=unew[IDX(i,j,ncols)]-u[IDX(i,j,ncols)];
                local_err+=diff*diff;
            }
        }
        MPI_Allreduce(&local_err,&global_err,1,MPI_DOUBLE,MPI_SUM,cart);
        global_err=sqrt(global_err);
        double *tmp=u;u=unew;unew=tmp;
        iter++;
    }
    if(rank==0)printf("%d %e\n",iter,global_err);
    free(u);free(unew);free(f);
    MPI_Type_free(&coltype);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
