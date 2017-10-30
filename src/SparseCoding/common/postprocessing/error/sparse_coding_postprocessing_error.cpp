#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../sparse_coding.h"

void SparseCoding::sparse_coding_reconstruction_error() {
    LOGR("=========== sparse_coding_reconstruction_error ===========", sparse_context.my_rank, sparse_context.master);

    float *pinv_pod_modes_into_sparse_modes = NULL, *pod_modes_into_sparse_modes = NULL, *pod_modes_into_sparse_modes_dup = NULL, *corrected_sparse_coeff = NULL, *recon_snapshots_from_sparse_modes = NULL;
    int rows_local_pod_modes_into_sparse_modes, cols_local_pod_modes_into_sparse_modes, rows_local_pinv_pod_modes_into_sparse_modes, cols_local_pinv_pod_modes_into_sparse_modes;
    int rows_local_corrected_sparse_coeff, cols_local_corrected_sparse_coeff, rows_local_recon_snapshots_from_sparse_modes, cols_local_recon_snapshots_from_sparse_modes;
    float norm_numerator, norm_denominator;

    int truncated_grid_points_in_all_dim = 3 * sparse_context.truncated_grid_points;

    matrix_mul_1D_process_grid(sparse_context.pod_bases, sparse_context.sparse_modes, &pod_modes_into_sparse_modes, rows_local_pod_modes_into_sparse_modes, cols_local_pod_modes_into_sparse_modes,\
                               truncated_grid_points_in_all_dim, sparse_context.rank_eigen_values, sparse_context.num_modes, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== pod_modes_into_sparse_modes ============================");
    display(pod_modes_into_sparse_modes, rows_local_pod_modes_into_sparse_modes * cols_local_pod_modes_into_sparse_modes);

    allocate(&pod_modes_into_sparse_modes_dup, rows_local_pod_modes_into_sparse_modes * cols_local_pod_modes_into_sparse_modes);
    duplicate(pod_modes_into_sparse_modes, pod_modes_into_sparse_modes_dup, rows_local_pod_modes_into_sparse_modes * cols_local_pod_modes_into_sparse_modes);

    // deallocating right after use to reduce peak mem usage
    deallocate(&sparse_context.pod_bases);

    pseudo_inverse(pod_modes_into_sparse_modes_dup, &pinv_pod_modes_into_sparse_modes, truncated_grid_points_in_all_dim, sparse_context.num_modes, \
                   rows_local_pinv_pod_modes_into_sparse_modes, cols_local_pinv_pod_modes_into_sparse_modes);

    deallocate(&pod_modes_into_sparse_modes_dup);

    LOG("=============== pinv_pod_modes_into_sparse_modes ============================");
    display(pinv_pod_modes_into_sparse_modes, rows_local_pinv_pod_modes_into_sparse_modes * cols_local_pinv_pod_modes_into_sparse_modes);

    matrix_mul_1D_process_grid(pinv_pod_modes_into_sparse_modes, truncated_snapshots, &corrected_sparse_coeff, rows_local_corrected_sparse_coeff, cols_local_corrected_sparse_coeff,\
                               sparse_context.num_modes, truncated_grid_points_in_all_dim, sparse_context.num_snapshots, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== corrected_sparse_coeff ============================");
    display(corrected_sparse_coeff, rows_local_corrected_sparse_coeff * cols_local_corrected_sparse_coeff);

    // deallocating right after use to reduce peak mem usage
    deallocate(&pinv_pod_modes_into_sparse_modes);

    matrix_mul_1D_process_grid(pod_modes_into_sparse_modes, corrected_sparse_coeff, &recon_snapshots_from_sparse_modes, rows_local_recon_snapshots_from_sparse_modes, cols_local_recon_snapshots_from_sparse_modes,\
                               truncated_grid_points_in_all_dim, sparse_context.num_modes, sparse_context.num_snapshots, 'N', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== recon_snapshots_from_sparse_modes ============================");
    display(recon_snapshots_from_sparse_modes, rows_local_recon_snapshots_from_sparse_modes * cols_local_recon_snapshots_from_sparse_modes);

    // deallocating right after use to reduce peak mem usage
    deallocate(&pod_modes_into_sparse_modes);
    deallocate(&corrected_sparse_coeff);

    allocate(&sparse_context.sparse_coding_rms_error, sparse_context.snapshots_per_rank);
    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        unsigned long stride = i * truncated_grid_points_in_all_dim;
        for (int j = 0; j < truncated_grid_points_in_all_dim; j++) {
            recon_snapshots_from_sparse_modes[stride + j] = truncated_snapshots[stride + j] - recon_snapshots_from_sparse_modes[stride + j];
        }

        norm_numerator = norm(&recon_snapshots_from_sparse_modes[stride + j], truncated_grid_points_in_all_dim);
        norm_denominator = norm(&truncated_snapshots[stride + j], truncated_grid_points_in_all_dim);
        sparse_context.sparse_coding_rms_error[i] = (float(100) * norm_numerator) / norm_denominator;
    }

    LOG("=============== sparse_coding_rms_error ============================");
    display(sparse_context.sparse_coding_rms_error, sparse_context.snapshots_per_rank);

    deallocate(&recon_snapshots_from_sparse_modes);
}
