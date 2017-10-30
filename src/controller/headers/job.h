#ifndef JOB_H
#define JOB_H

#include "../../common/headers/context.h"
#include "../../common/headers/mpi_context.h"

typedef enum {
    PODSerial,
    PODParallelRowCyclic,
    PODParallelColCyclic,
    PODReconstructCoefficientsFromModes,
    SparseCodingSerialDenseMatrix,
    SparseCodingSerialSparseMatrix,
    SparseCodingParallelDenseMatrix
} JobType;

class Job {
  public:
    static void create_pod_job(Context& context, MPIContext& mpi_context);
    static void create_sparse_coding_job(Context& context, MPIContext& mpi_context);
    static void start_pod_serial_job(Context& context, MPIContext& mpi_context);
    static void start_pod_parallel_col_cyclic_job(Context& context, MPIContext& mpi_context);
    static void start_pod_parallel_row_cyclic_job(Context& context, MPIContext& mpi_context);
    static void start_pod_reconstruct_coefficients_from_modes(Context& context, MPIContext& mpi_context);
    static void start_sparse_coding_serial_dense_matrix_job(Context& context, MPIContext& mpi_context);
    static void start_sparse_coding_serial_sparse_matrix_job(Context& context, MPIContext& mpi_context);
    static void start_sparse_coding_parallel_dense_matrix_job(Context& context, MPIContext& mpi_context);
};


#endif
