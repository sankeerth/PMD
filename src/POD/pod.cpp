#include "../common/headers/log.h"
#include "../common/headers/utilities.h"
#include "../common/headers/scalapack_helpers.h"
#include "pod.h"

POD::POD(const Context& context, const MPIContext& pod_context)
    : pod_context(context, pod_context)
{
    if (pod_context.my_rank == pod_context.master) {
        start_time = MPI_Wtime();
    }
}

void POD::mesh_processing() {
    if (pod_context.my_rank == pod_context.master) {
        create_snapshots_list();
        read_mesh(false);
        grid_metrics();
        compute_mesh_volume();
    }

    measure_time_for_function(start_time, "Functions related to Mesh, done only by master", pod_context.my_rank, pod_context.master);
}

void POD::snapshots_preprocessing_1D_procs_along_col() {
    MPI_Barrier(MPI_COMM_WORLD);
    distribute_snapshot_filenames();

    MPI_Barrier(MPI_COMM_WORLD);
    make_snapshot_truncated_indices();

    MPI_Barrier(MPI_COMM_WORLD);
    read_truncated_snapshots_1D_procs_along_col();
    measure_time_for_function(start_time, "Read snapshots done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_average_1D_procs_along_col();
    measure_time_for_function(start_time, "Compute average done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_fluctuating_component_1D_procs_along_col();
    measure_time_for_function(start_time, "Compute fluctuating component done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    distribute_truncated_vol_grid();

    MPI_Barrier(MPI_COMM_WORLD);
    compute_snapshot_matrix_1D_procs_along_col();
    measure_time_for_function(start_time, "Compute snapshot matrix done", pod_context.my_rank, pod_context.master);
}

void POD::snapshots_preprocessing_1D_procs_along_row() {
    MPI_Barrier(MPI_COMM_WORLD);
    distribute_snapshot_filenames();

    MPI_Barrier(MPI_COMM_WORLD);
    make_snapshot_truncated_indices();

    MPI_Barrier(MPI_COMM_WORLD);
    read_truncated_snapshots_1D_procs_along_row();
    measure_time_for_function(start_time, "Read snapshots done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_average_1D_procs_along_row();
    measure_time_for_function(start_time, "Compute average done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_fluctuating_component_1D_procs_along_row();
    measure_time_for_function(start_time, "Compute fluctuating component done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    distribute_truncated_vol_grid();

    MPI_Barrier(MPI_COMM_WORLD);
    compute_snapshot_matrix_1D_procs_along_row();
    measure_time_for_function(start_time, "Compute snapshot matrix done", pod_context.my_rank, pod_context.master);
}

void POD::compute_pod_1D_procs_along_col() {
    MPI_Barrier(MPI_COMM_WORLD);
    compute_covariance_matrix();
    measure_time_for_function(start_time, "Compute covariance matrix done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_svd();
    measure_time_for_function(start_time, "Compute SVD done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_rank_eigen_values();

    MPI_Barrier(MPI_COMM_WORLD);
    modify_left_singular_matrix_1D_procs_along_col();
    measure_time_for_function(start_time, "Modify left singular matrix done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_pod_modes();
    measure_time_for_function(start_time, "Compute POD modes done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_pod_coefficients();
    measure_time_for_function(start_time, "Compute POD coefficients done", pod_context.my_rank, pod_context.master);
}

void POD::compute_pod_1D_procs_along_row() {
    MPI_Barrier(MPI_COMM_WORLD);
    compute_covariance_matrix();
    measure_time_for_function(start_time, "Compute covariance matrix done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_svd();
    measure_time_for_function(start_time, "Compute SVD done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_rank_eigen_values();

    MPI_Barrier(MPI_COMM_WORLD);
    modify_left_singular_matrix_1D_procs_along_row();
    measure_time_for_function(start_time, "Modify left singular matrix done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_pod_modes();
    measure_time_for_function(start_time, "Compute POD modes done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_pod_coefficients();
    measure_time_for_function(start_time, "Compute POD coefficients done", pod_context.my_rank, pod_context.master);
}

void POD::compute_pod_error_1D_procs_along_col() {
    MPI_Barrier(MPI_COMM_WORLD);
    pod_reconstruction_error();
    measure_time_for_function(start_time, "Compute reconstruction error done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    pod_rms_error_1D_procs_along_col();
    measure_time_for_function(start_time, "Compute POD RMS error done", pod_context.my_rank, pod_context.master);

    deallocate_snapshots_recon_errror();
}

void POD::compute_pod_error_1D_procs_along_row() {
    MPI_Barrier(MPI_COMM_WORLD);
    pod_reconstruction_error();
    measure_time_for_function(start_time, "Compute reconstruction error done", pod_context.my_rank, pod_context.master);

    MPI_Barrier(MPI_COMM_WORLD);
    pod_rms_error_1D_procs_along_row();
    measure_time_for_function(start_time, "Compute POD RMS error done", pod_context.my_rank, pod_context.master);

    deallocate_snapshots_recon_errror();
}

void POD::write_pod_output_files_1D_procs_along_col() {
    if (pod_context.is_write_coefficients_and_error_to_binary_format) {
        MPI_Barrier(MPI_COMM_WORLD);
        write_pod_coefficients_binary_1D_procs_along_col();
        measure_time_for_function(start_time, "Writing POD coefficients done", pod_context.my_rank, pod_context.master);

        MPI_Barrier(MPI_COMM_WORLD);
        write_pod_rms_error_binary_1D_procs_along_col();
        measure_time_for_function(start_time, "Writing POD RMS error done", pod_context.my_rank, pod_context.master);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (pod_context.is_write_pod_modes_to_binary_format) {
        write_pod_modes_binary_1D_procs_along_col();
        measure_time_for_function(start_time, "Writing POD modes done", pod_context.my_rank, pod_context.master);
    }
}

void POD::write_pod_output_files_1D_procs_along_row() {
    if (pod_context.is_write_coefficients_and_error_to_binary_format) {
        MPI_Barrier(MPI_COMM_WORLD);
        write_pod_coefficients_binary_1D_procs_along_row();
        measure_time_for_function(start_time, "Writing POD coefficients done", pod_context.my_rank, pod_context.master);

        if (pod_context.my_rank == pod_context.master) {
            write_pod_rms_error_binary_1D_procs_along_row();
            measure_time_for_function(start_time, "Writing POD RMS error done", pod_context.my_rank, pod_context.master);
        }
    }

    if (pod_context.is_write_pod_modes_to_binary_format) {
        MPI_Barrier(MPI_COMM_WORLD);
        write_pod_modes_binary_1D_procs_along_row();
        measure_time_for_function(start_time, "Writing POD modes done", pod_context.my_rank, pod_context.master);
    }
}
