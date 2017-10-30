#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../pod.h"

void POD::pod_reconstruction_error() {
    LOGR("=========== pod_reconstruction_error ===========", pod_context.my_rank, pod_context.master);

    LOGD("num_modes_in_my_rank", pod_context.num_modes_in_my_rank);

    int rows_local_reconstruction_error, cols_local_reconstruction_error;

    int m = pod_context.truncated_grid_points_in_all_dim;
    int n = pod_context.num_modes;

    matrix_mul_1D_process_grid(pod_context.pod_bases, pod_context.pod_coefficients, &pod_context.pod_reconstruction_error, rows_local_reconstruction_error, cols_local_reconstruction_error,\
                               m, n, pod_context.num_snapshots, 'N', 'N', pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, false);

    LOG("=============== reconstruction error ============================");
    display(pod_context.pod_reconstruction_error, rows_local_reconstruction_error * cols_local_reconstruction_error);
}
