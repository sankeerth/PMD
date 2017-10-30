#include <math.h>
#include "../../../common/headers/log.h"
#include "../../pod.h"

void POD::compute_average_1D_procs_along_col() {
    LOGR("=========== compute_average_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    allocate(&pod_context.average, pod_context.truncated_grid_points_in_all_dim);
    initialize(pod_context.average,  pod_context.truncated_grid_points_in_all_dim);

    unsigned long stride = 0;
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        stride = i * pod_context.truncated_grid_points_in_all_dim;
        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            pod_context.average[j] += pod_context.truncated_snapshots[stride + j];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, pod_context.average, pod_context.truncated_grid_points_in_all_dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    display(pod_context.average, pod_context.truncated_grid_points_in_all_dim);
}

void POD::compute_fluctuating_component_1D_procs_along_col() {
    LOGR("=========== compute_fluctuating_component ===========", pod_context.my_rank, pod_context.master);

    unsigned long stride = 0;
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        stride = i * pod_context.truncated_grid_points_in_all_dim;
        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            pod_context.truncated_snapshots[stride +j] = pod_context.truncated_snapshots[stride + j] - (pod_context.average[j] / pod_context.num_snapshots);
        }
        display(&pod_context.truncated_snapshots[stride], pod_context.truncated_grid_points_in_all_dim);
        NEWLINE;
    }

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.average);
}

void POD::compute_snapshot_matrix_1D_procs_along_col() {
    LOGR("=========== compute_snapshot_matrix ===========", pod_context.my_rank, pod_context.master);

    unsigned long stride = 0;
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        stride = i * pod_context.truncated_grid_points_in_all_dim;
        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            pod_context.truncated_snapshots[stride + j] = pod_context.truncated_snapshots[stride + j] * pod_context.truncated_vol_grid[j % pod_context.truncated_grid_points];
        }
        display(&pod_context.truncated_snapshots[stride], pod_context.truncated_grid_points_in_all_dim);
    }

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.truncated_vol_grid);
}

void POD::modify_left_singular_matrix_1D_procs_along_col() {
    LOGR("=========== modify_left_singular_matrix ===========", pod_context.my_rank, pod_context.master);

    // divide each column of left singular matrix by sqrt(num_snapshots * eigen_value(i))
    // This is need to compute pod_bases, however, since pod_bases is huge and of the same size of truncated snapshots
    // it is better to do the division on left singular vectors inplace and the multiply left singular matrix with truncated snapshots

    unsigned long stride;
    float eigen_value_product_num_snapshots;

    // TODO: there could be a bug here with the usage of snapshots_per_rank for odd num of procs
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        eigen_value_product_num_snapshots = 1 / sqrt(pod_context.num_snapshots * pod_context.eigen_values[pod_context.index_of_snapshot_filenames[i]]);
        stride = i * pod_context.num_snapshots;
        for (unsigned long j = 0; j < pod_context.num_snapshots; j++) {
            pod_context.left_singular_vectors[stride + j] = pod_context.left_singular_vectors[stride + j] * eigen_value_product_num_snapshots;
        }
    }

    LOG("=============== Modified left singular vector ============================");
    display(pod_context.left_singular_vectors, pod_context.size_of_left_singular_vectors);
}
