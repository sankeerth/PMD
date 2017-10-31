#include <math.h>
#include "../../../common/headers/log.h"
#include "../../pod.h"

void POD::compute_average_1D_procs_along_row() {
    LOGR("=========== compute_average_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    allocate(&pod_context.average, pod_context.num_snapshot_points_per_proc);
    initialize(pod_context.average, pod_context.num_snapshot_points_per_proc);

    for (unsigned long i = 0; i < pod_context.num_snapshots; i++) {
        for (unsigned long j = 0; j < pod_context.num_snapshot_points_per_proc; j++) {
            pod_context.average[j] += pod_context.truncated_snapshots[i * pod_context.num_snapshot_points_per_proc + j];
        }
    }

    for (unsigned long i = 0; i < pod_context.num_snapshot_points_per_proc; i++) {
        pod_context.average[i] = pod_context.average[i] / pod_context.num_snapshots;
    }

    LOG("============ average ===========");
    display(pod_context.average, pod_context.num_snapshot_points_per_proc);
}

void POD::compute_fluctuating_component_1D_procs_along_row() {
    LOGR("=========== compute_fluctuating_component_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    unsigned long stride = 0;
    for (int i = 0; i < pod_context.num_snapshots; i++) {
        stride = i * pod_context.num_snapshot_points_per_proc;
        for (unsigned long j = 0; j < pod_context.num_snapshot_points_per_proc; j++) {
            pod_context.truncated_snapshots[stride + j] = pod_context.truncated_snapshots[stride + j] - pod_context.average[j];
        }
    }

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.average);
}

void POD::compute_snapshot_matrix_1D_procs_along_row() {
    LOGR("=========== compute_snapshot_matrix_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    unsigned long stride = 0;

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        stride = i * pod_context.num_snapshot_points_per_proc;
        for (unsigned long j = 0, current_index = pod_context.my_rank; j < pod_context.num_snapshot_points_per_proc; j++, current_index += pod_context.num_procs) {
            pod_context.truncated_snapshots[stride + j] = pod_context.truncated_snapshots[stride + j] * pod_context.truncated_vol_grid[current_index % pod_context.truncated_grid_points];
        }
    }

    // freeing memory right after use to reduce peak memory usage
    deallocate(&pod_context.truncated_vol_grid);
}

void POD::modify_left_singular_matrix_1D_procs_along_row() {
    LOGR("=========== modify_left_singular_matrix_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    unsigned long stride = 0;
    float eigen_value_product_num_snapshots;

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        stride = i * pod_context.snapshots_per_rank;
        eigen_value_product_num_snapshots = 1 / sqrt(pod_context.num_snapshots * pod_context.eigen_values[i]);
        for (int j = 0; j < pod_context.snapshots_per_rank; j++) {
            pod_context.left_singular_vectors[stride + j] = pod_context.left_singular_vectors[stride + j] * eigen_value_product_num_snapshots;
        }
    }

    LOG("=============== Modified left singular vector ============================");
    display(pod_context.left_singular_vectors, pod_context.size_of_left_singular_vectors);
}

