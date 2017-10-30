#include "sparse_coding.h"

SparseCoding::SparseCoding(const Context& context, const MPIContext& mpi_context)
    : sparse_context(context, mpi_context)
{ }
