#include "sparse_coding.h"

SparseCoding::SparseCoding(const Context& context, const MPIContext& mpi_context)
    : sparse_context(context, mpi_context)
{
    if (sparse_context.my_rank == sparse_context.master) {
        start_time = MPI_Wtime();
    }
}

void SparseCoding::sparse_coding_preprocessing() {
    read_eigen_values();
    read_pod_coefficients();
    read_pod_coefficients_dummy();
    initial_guess_sparse_modes();
    normalize_sparse_modes();
    initialize_eplison_convergence_value();
}

void SparseCoding::compute_sparse_coding_parallel() {
    generate_sparse_modes_parallel();
}

void SparseCoding::write_sparse_coding_output_files_parallel() {

    if(sparse_context.is_write_sparse_transformation_matrix) {
        write_sparse_modes_binary();
    }

    if(sparse_context.is_write_sparse_coefficients) {
        write_sparse_coefficients_binary_parallel();
    }

    if(sparse_context.is_write_sparse_modes_in_original_domain) {
        write_sparse_modes_in_original_domain_binary();
    }

    if(sparse_context.is_write_corrected_sparse_coefficients) {
        write_corrected_sparse_coefficients_binary();
    }

    if (sparse_context.is_write_sparse_reconstruction_error) {
        write_sparse_coding_rms_error_binary();
    }
}

void SparseCoding::write_sparse_coding_output_files_serial() {

    if(sparse_context.is_write_sparse_transformation_matrix) {
        write_sparse_modes_binary();
    }

    if(sparse_context.is_write_sparse_coefficients) {
        write_sparse_coefficients_binary_serial();
    }

    if(sparse_context.is_write_sparse_modes_in_original_domain) {
        write_sparse_modes_in_original_domain_binary();
    }

    if(sparse_context.is_write_corrected_sparse_coefficients) {
        write_corrected_sparse_coefficients_binary();
    }

    if (sparse_context.is_write_sparse_reconstruction_error) {
        write_sparse_coding_rms_error_binary();
    }
}

void SparseCoding::compute_sparse_coding_serial() {
    generate_sparse_modes_serial();
}
