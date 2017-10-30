#include "../../../common/headers/log.h"
#include "../../../common/headers/utilities.h"
#include "../../../common/headers/scalapack_helpers.h"
#include "../../pod.h"

void POD::compute_covariance_matrix() {
    LOGR("=========== compute_covariance_matrix ===========", pod_context.my_rank, pod_context.master);

    int rows_local_covariance_matrix, cols_local_covariance_matrix;

    int m = pod_context.truncated_grid_points_in_all_dim;
    int n = pod_context.num_snapshots;

    matrix_mul_1D_process_grid(pod_context.truncated_snapshots, pod_context.truncated_snapshots, &pod_context.covariance_matrix, rows_local_covariance_matrix, cols_local_covariance_matrix,\
                               m, n, n, 'T', 'N', pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, true);

    // divide the covariance matrix by number of snapshots
    for (int i = 0; i < rows_local_covariance_matrix * cols_local_covariance_matrix; i++) {
        pod_context.covariance_matrix[i] = pod_context .covariance_matrix[i] / pod_context.num_snapshots;
    }

    LOG("=============== covariance matrix ============================");
    display(pod_context.covariance_matrix, rows_local_covariance_matrix * cols_local_covariance_matrix);
}

void POD::compute_svd() {
    LOGR("=========== compute_svd ===========", pod_context.my_rank, pod_context.master);

    int size_of_eigen_values, rows_U_local, cols_U_local, rows_Vt_local, cols_Vt_local;
    float *Vt = NULL;

    singular_value_decomposition(pod_context.covariance_matrix, 'V', 'N', pod_context.num_snapshots, pod_context.num_snapshots, &pod_context.eigen_values, &pod_context.left_singular_vectors, &Vt, size_of_eigen_values, rows_U_local, cols_U_local, rows_Vt_local, cols_Vt_local,\
                                 pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, 1, 1);

    pod_context.size_of_eigen_values = size_of_eigen_values;
    pod_context.size_of_left_singular_vectors = rows_U_local * cols_U_local;

    LOG("=============== eigen values ============================");
    display(pod_context.eigen_values, size_of_eigen_values);
    LOG("=============== Left singular vector ============================");
    display(pod_context.left_singular_vectors, rows_U_local * cols_U_local);

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.covariance_matrix);
}

void POD::compute_rank_eigen_values() {
    LOGR("=========== compute_rank_eigen_values ===========", pod_context.my_rank, pod_context.master);

    float *eigen_values_copy = NULL;
    float max_eigen_value = pod_context.eigen_values[0];

    allocate(&eigen_values_copy, pod_context.size_of_eigen_values);
    duplicate(pod_context.eigen_values, eigen_values_copy, pod_context.size_of_eigen_values);

    for (int i = 0; i < pod_context.size_of_eigen_values; i++) {
        eigen_values_copy[i] = eigen_values_copy[i] / max_eigen_value;
    }

    for (int i = 0; i < pod_context.size_of_eigen_values; i++) {
        if (eigen_values_copy[i] > pod_context.epsilon_rank) {
            pod_context.rank_eigen_values += 1;
        } else {
            break;
        }
    }

    LOGD("rank of snapshot matrix", pod_context.rank_eigen_values);

    if (pod_context.num_modes > pod_context.rank_eigen_values) {
        LOG("Number of modes can not be larger than the rank of the snapshot matrix");
        LOG("Re run the calculation of POD with number of modes lesser or equal to the rank of the snapshots matrix");
    }
}

void POD::compute_pod_modes() {
    LOGR("=========== compute_pod_modes ===========", pod_context.my_rank, pod_context.master);

    int rows_local_pod_bases, cols_local_pod_bases;
    int num_pod_modes = MIN(pod_context.num_modes, pod_context.rank_eigen_values);

    matrix_mul_1D_process_grid(pod_context.truncated_snapshots, pod_context.left_singular_vectors, &pod_context.pod_bases, rows_local_pod_bases, cols_local_pod_bases,\
                               pod_context.truncated_grid_points_in_all_dim, pod_context.num_snapshots, num_pod_modes, 'N', 'N', pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, false);

    LOG("=============== POD modes ============================");
    display(pod_context.pod_bases, rows_local_pod_bases * cols_local_pod_bases);

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.left_singular_vectors);
}

void POD::compute_pod_coefficients() {
    LOGR("=========== compute_pod_coefficients ===========", pod_context.my_rank, pod_context.master);

    int rows_local_pod_coefficients, cols_local_pod_coefficients;
    int num_coefficients_per_file = MIN(pod_context.num_modes, pod_context.rank_eigen_values);

    matrix_mul_1D_process_grid(pod_context.pod_bases, pod_context.truncated_snapshots, &pod_context.pod_coefficients, rows_local_pod_coefficients, cols_local_pod_coefficients,\
                               pod_context.truncated_grid_points_in_all_dim, num_coefficients_per_file, pod_context.num_snapshots, 'T', 'N', pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, false);

    LOG("=============== POD coefficients ============================");
    display(pod_context.pod_coefficients, rows_local_pod_coefficients * cols_local_pod_coefficients);
}

void POD::compute_pod_coefficients_from_modes_and_snapshots() {
    LOGR("=========== compute_pod_coefficients_from_modes_and_snapshots ===========", pod_context.my_rank, pod_context.master);

    int rows_local_pod_coefficients_matrix, cols_local_pod_coefficients_matrix;

    int m = pod_context.truncated_grid_points_in_all_dim;
    int n = pod_context.num_snapshots;

    matrix_mul_1D_process_grid(pod_context.pod_bases, pod_context.truncated_snapshots, &pod_context.pod_coefficients, rows_local_pod_coefficients_matrix, cols_local_pod_coefficients_matrix,\
                               m, pod_context.num_modes, n, 'T', 'N', pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1, true);

    LOG("=============== POD coefficients ============================");
    display(pod_context.pod_coefficients, rows_local_pod_coefficients_matrix * cols_local_pod_coefficients_matrix);
}
