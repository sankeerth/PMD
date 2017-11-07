#include "sparse_coding.h"

SparseCoding::SparseCoding(const Context& context, const MPIContext& mpi_context)
    : sparse_context(context, mpi_context)
{ }

void SparseCoding::sparse_coding_preprocessing() {
    read_eigen_values();
    read_pod_coefficients();
    read_pod_coefficients_dummy();
    initial_guess_sparse_modes();
    normalize_sparse_modes();
    initialize_eplison_convergence_value();
}

void SparseCoding::compute_sparse_coding() {
    generate_sparse_modes_parallel();
}

void SparseCoding::sparse_coding_postprocessing() {

}

void SparseCoding::write_sparse_coding_output_files() {

    if(sparse_context.is_write_sparse_transformation_matrix) {
        write_sparse_modes_binary();
    }

    if(sparse_context.is_write_sparse_coefficients) {
        write_sparse_coefficients_binary();
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
