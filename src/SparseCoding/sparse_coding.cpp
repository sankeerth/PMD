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

void SparseCoding::compute_sparse_coding_error(float *truncated_snapshots) {
    read_pod_modes();
    sparse_coding_reconstruction_error(truncated_snapshots);
}

void SparseCoding::write_sparse_coding_output_files() {
    write_sparse_modes_binary();
    write_sparse_coefficients_binary();
    if (sparse_context.compute_sparse_coding_reconstruction_error) {
        write_sparse_coding_rms_error_binary();
    }
}
