#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../sparse_coding.h"


void SparseCoding::compute_sparse_modes_in_original_domain() {
    LOGR("=========== compute_sparse_modes_in_original_domain ===========", sparse_context.my_rank, sparse_context.master);


    int rows_local_pod_modes_into_sparse_modes, cols_local_pod_modes_into_sparse_modes;
    read_pod_modes();

    matrix_mul_1D_process_grid(sparse_context.pod_bases, sparse_context.sparse_modes, &sparse_context.pod_modes_into_sparse_modes, rows_local_pod_modes_into_sparse_modes, cols_local_pod_modes_into_sparse_modes,\
                               sparse_context.truncated_grid_points_in_all_dim, sparse_context.rank_eigen_values, sparse_context.num_modes, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== pod_modes_into_sparse_modes ============================");
    display(sparse_context.pod_modes_into_sparse_modes, rows_local_pod_modes_into_sparse_modes * cols_local_pod_modes_into_sparse_modes);

    // deallocating right after use to reduce peak mem usage
    deallocate(&sparse_context.pod_bases);

    if(!sparse_context.is_write_sparse_transformation_matrix)
        deallocate(&sparse_context.sparse_modes);

    return;
}


void SparseCoding::compute_corrected_sparse_coefficients(float *truncated_snapshots) {
    LOGR("=========== compute_corrected_sparse_coefficients ===========", sparse_context.my_rank, sparse_context.master);

    float *pod_modes_into_sparse_modes_dup = NULL;
    float *pinv_pod_modes_into_sparse_modes = NULL;
    int  rows_local_pinv_pod_modes_into_sparse_modes, cols_local_pinv_pod_modes_into_sparse_modes;
    int rows_local_corrected_sparse_coeff, cols_local_corrected_sparse_coeff;

    allocate(&pod_modes_into_sparse_modes_dup, sparse_context.truncated_grid_points_in_all_dim * sparse_context.num_modes_in_my_rank);
    duplicate(sparse_context.pod_modes_into_sparse_modes, pod_modes_into_sparse_modes_dup, sparse_context.truncated_grid_points_in_all_dim * sparse_context.num_modes_in_my_rank);



    pseudo_inverse(pod_modes_into_sparse_modes_dup, &pinv_pod_modes_into_sparse_modes, sparse_context.truncated_grid_points_in_all_dim, sparse_context.num_modes, \
                   rows_local_pinv_pod_modes_into_sparse_modes, cols_local_pinv_pod_modes_into_sparse_modes);

    deallocate(&pod_modes_into_sparse_modes_dup);

    LOG("=============== pinv_pod_modes_into_sparse_modes ============================");
    display(pinv_pod_modes_into_sparse_modes, rows_local_pinv_pod_modes_into_sparse_modes * cols_local_pinv_pod_modes_into_sparse_modes);

    matrix_mul_1D_process_grid(pinv_pod_modes_into_sparse_modes, truncated_snapshots, &sparse_context.corrected_sparse_coeff, rows_local_corrected_sparse_coeff, cols_local_corrected_sparse_coeff,\
                               sparse_context.num_modes, sparse_context.truncated_grid_points_in_all_dim, sparse_context.num_snapshots, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== corrected_sparse_coeff ============================");
    display(sparse_context.corrected_sparse_coeff, rows_local_corrected_sparse_coeff * cols_local_corrected_sparse_coeff);

    // deallocating right after use to reduce peak mem usage
    deallocate(&pinv_pod_modes_into_sparse_modes);
}


void SparseCoding::sparse_coding_reconstruction_error(float *truncated_snapshots) {
    LOGR("=========== sparse_coding_reconstruction_error ===========", sparse_context.my_rank, sparse_context.master);

    float  *recon_snapshots_from_sparse_modes = NULL;
    int  rows_local_recon_snapshots_from_sparse_modes, cols_local_recon_snapshots_from_sparse_modes;
    float norm_numerator, norm_denominator;


    matrix_mul_1D_process_grid(sparse_context.pod_modes_into_sparse_modes, sparse_context.corrected_sparse_coeff, &recon_snapshots_from_sparse_modes, rows_local_recon_snapshots_from_sparse_modes, cols_local_recon_snapshots_from_sparse_modes,\
                               sparse_context.truncated_grid_points_in_all_dim, sparse_context.num_modes, sparse_context.num_snapshots, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== recon_snapshots_from_sparse_modes ============================");
    display(recon_snapshots_from_sparse_modes, rows_local_recon_snapshots_from_sparse_modes * cols_local_recon_snapshots_from_sparse_modes);

    allocate(&sparse_context.sparse_coding_rms_error, sparse_context.snapshots_per_rank);
    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        unsigned long stride = i * sparse_context.truncated_grid_points_in_all_dim;
        for (int j = 0; j < sparse_context.truncated_grid_points_in_all_dim; j++) {
            recon_snapshots_from_sparse_modes[stride + j] = truncated_snapshots[stride + j] - recon_snapshots_from_sparse_modes[stride + j];
        }

        norm_numerator = norm(&recon_snapshots_from_sparse_modes[stride + j], sparse_context.truncated_grid_points_in_all_dim);
        norm_denominator = norm(&truncated_snapshots[stride + j], sparse_context.truncated_grid_points_in_all_dim);
        sparse_context.sparse_coding_rms_error[i] = (float(100) * norm_numerator) / norm_denominator;
    }

    LOG("=============== sparse_coding_rms_error ============================");
    display(sparse_context.sparse_coding_rms_error, sparse_context.snapshots_per_rank);

    deallocate(&recon_snapshots_from_sparse_modes);
}
