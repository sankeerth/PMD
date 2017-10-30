#include "../../common/headers/log.h"
#include "../headers/job.h"
#include "../../POD/pod.h"
#include "../../SparseCoding/sparse_coding.h"

void Job::create_pod_job(Context& context, MPIContext& mpi_context) {
    switch(context.job) {
        case PODJob::PODSerial:
            Job::start_pod_serial_job(context, mpi_context);
            break;

        case PODJob::PODParallelColCyclic:
            Job::start_pod_parallel_col_cyclic_job(context, mpi_context);
            break;

        case PODJob::PODParallelRowCyclic:
            Job::start_pod_parallel_row_cyclic_job(context, mpi_context);
            break;

        case PODJob::PODReconstructCoefficientsFromModes:
            Job::start_pod_reconstruct_coefficients_from_modes(context, mpi_context);

        default:
            LOGR("Wrong Job provided. Please provide job '0' for PODSerial, '1' for PODParallelColCyclic, '2' for PODParallelRowCyclic and '3' for PODReconstructCoefficientsFromModes",\
                 mpi_context.my_rank, mpi_context.master);
            break;
    }
}

void Job::create_sparse_coding_job(Context &context, MPIContext &mpi_context) {
    switch (context.job) {
        case SparseCodingJob::SparseCodingSerialDenseMatrix:
            start_sparse_coding_serial_dense_matrix_job(context, mpi_context);
            break;

        case SparseCodingJob::SparseCodingSerialSparseMatrix:
            start_sparse_coding_serial_sparse_matrix_job(context, mpi_context);
            break;

        case SparseCodingJob::SparseCodingParallelDenseMatrix:
            start_sparse_coding_parallel_dense_matrix_job(context, mpi_context);
            break;

        default:
            LOGR("Wrong Job provided. Please provide job '0' for SparseCodingSerialDenseMatrix, '1' for SparseCodingSerialSparseMatrix and '2' for SparseCodingParallelDenseMatrix",\
                 mpi_context.my_rank, mpi_context.master);
            break;
    }
}

void Job::start_pod_serial_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_pod_serial_job ***********", mpi_context.my_rank, mpi_context.master);

    POD pod(context, mpi_context);

    pod.mesh_processing();
    pod.snapshots_preprocessing_1D_procs_along_col();
    pod.compute_pod_1D_procs_along_col();
    pod.compute_pod_error_1D_procs_along_col();
    pod.write_pod_output_files_1D_procs_along_col();
    pod.cleanup_memory();
}

void Job::start_pod_parallel_col_cyclic_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_pod_parallel_col_cyclic_job ***********", mpi_context.my_rank, mpi_context.master);

    POD pod(context, mpi_context);

    pod.mesh_processing();
    pod.snapshots_preprocessing_1D_procs_along_col();
    pod.compute_pod_1D_procs_along_col();
    pod.compute_pod_error_1D_procs_along_col();
    pod.write_pod_output_files_1D_procs_along_col();
    pod.cleanup_memory();
}

void Job::start_pod_parallel_row_cyclic_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_pod_parallel_row_cyclic_job ***********", mpi_context.my_rank, mpi_context.master);

    POD pod(context, mpi_context);

    pod.mesh_processing();
    pod.snapshots_preprocessing_1D_procs_along_row();
    pod.compute_pod_1D_procs_along_row();
    pod.compute_pod_error_1D_procs_along_row();
    pod.write_pod_output_files_1D_procs_along_row();
    pod.cleanup_memory();
}

void Job::start_pod_reconstruct_coefficients_from_modes(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_pod_reconstruct_coefficients_from_modes ***********", mpi_context.my_rank, mpi_context.master);

    POD pod(context, mpi_context);

    pod.mesh_processing();
    pod.make_snapshot_truncated_indices();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.distribute_snapshot_filenames();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.read_mean_flow_binary_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.read_truncated_snapshots_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.distribute_truncated_vol_grid();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.compute_fluctuating_component_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.compute_snapshot_matrix_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.read_pod_modes_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.compute_pod_coefficients_from_modes_and_snapshots();
    MPI_Barrier(MPI_COMM_WORLD);
    pod.write_pod_coefficients_binary_1D_procs_along_col();
    MPI_Barrier(MPI_COMM_WORLD);
}

void Job::start_sparse_coding_serial_dense_matrix_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_sparse_coding_serial_dense_matrix_job ***********", mpi_context.my_rank, mpi_context.master);

    SparseCoding sparse_coding(context, mpi_context);
}

void Job::start_sparse_coding_serial_sparse_matrix_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_sparse_coding_serial_sparse_matrix_job ***********", mpi_context.my_rank, mpi_context.master);

    SparseCoding sparse_coding(context, mpi_context);
}

void Job::start_sparse_coding_parallel_dense_matrix_job(Context &context, MPIContext &mpi_context) {
    LOGR("*********** start_sparse_coding_parallel_dense_matrix_job ***********", mpi_context.my_rank, mpi_context.master);

    SparseCoding sparse_coding(context, mpi_context);
}
