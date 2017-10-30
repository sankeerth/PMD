#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include<mpi.h>

class MPIContext {
  public:
    int my_rank;
    int master;
    int num_procs;
    int num_procs_along_row;
    int num_procs_along_col;

    MPIContext()
        : master(0)
        , my_rank(0)
        , num_procs(0)
        , num_procs_along_row(0)
        , num_procs_along_col(0)
    { }

    MPIContext(const MPIContext& mpi_context)
        : master(mpi_context.master)
        , my_rank(mpi_context.my_rank)
        , num_procs(mpi_context.num_procs)
        , num_procs_along_row(mpi_context.num_procs_along_row)
        , num_procs_along_col(mpi_context.num_procs_along_col)
    { }
};

#endif // MPI_CONTEXT_H
