#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../sparse_coding.h"

void SparseCoding::sparse_coding_reconstruction_error() {
    LOGR("=========== sparse_coding_reconstruction_error ===========", sparse_context.my_rank, sparse_context.master);

    float *pinv_sparse_modes = NULL, *pinv_sparse_modes_into_pod_modes = NULL;
    int rows_local_pinv_sparse_modes, cols_local_pinv_sparse_modes, rows_local_pinv_sparse_modes_into_pod_modes, cols_local_pinv_sparse_modes_into_pod_modes;

    pseudo_inverse(sparse_context.sparse_modes, &pinv_sparse_modes, sparse_context.rank_eigen_values, sparse_context.num_modes, rows_local_pinv_sparse_modes, cols_local_pinv_sparse_modes);

    display(pinv_sparse_modes, rows_local_pinv_sparse_modes * cols_local_pinv_sparse_modes);

    matrix_mul_1D_process_grid(pinv_sparse_modes, sparse_context.pod_bases, &pinv_sparse_modes_into_pod_modes, rows_local_pinv_sparse_modes_into_pod_modes, cols_local_pinv_sparse_modes_into_pod_modes,\
                               sparse_context.num_modes, 3 * sparse_context.truncated_grid_points, sparse_context.rank_eigen_values, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    display(pinv_sparse_modes_into_pod_modes, rows_local_pinv_sparse_modes_into_pod_modes * cols_local_pinv_sparse_modes_into_pod_modes);
}
