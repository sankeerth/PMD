#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::pod_rms_error_1D_procs_along_row() {
    LOGR("=========== compute_average_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    double *norm_numerator = NULL, *norm_denominator = NULL, *norm_numerator_reduce_in_master = NULL, *norm_denominator_reduce_in_master = NULL;
    unsigned long stride = 0;

    allocate(&norm_numerator, pod_context.num_snapshots);
    allocate(&norm_denominator, pod_context.num_snapshots);

    if (pod_context.my_rank == pod_context.master) {
        allocate(&norm_numerator_reduce_in_master, pod_context.num_snapshots);
        allocate(&norm_denominator_reduce_in_master, pod_context.num_snapshots);
    }

    initialize(norm_numerator, pod_context.num_snapshots);
    initialize(norm_denominator, pod_context.num_snapshots);

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        stride = i * pod_context.num_snapshot_points_per_proc;
        for (unsigned long j = 0; j < pod_context.num_snapshot_points_per_proc; j++) {
            norm_numerator[i] += ((double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j])) * ((double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j]));
            norm_denominator[i] += (double) pod_context.truncated_snapshots[stride + j] * (double) pod_context.truncated_snapshots[stride + j];
        }
    }

    MPI_Reduce(norm_numerator, norm_numerator_reduce_in_master, pod_context.num_snapshots, MPI_DOUBLE, MPI_SUM, pod_context.master, MPI_COMM_WORLD);
    MPI_Reduce(norm_denominator, norm_denominator_reduce_in_master, pod_context.num_snapshots, MPI_DOUBLE, MPI_SUM, pod_context.master, MPI_COMM_WORLD);

    if (pod_context.my_rank == pod_context.master) {
        LOG("=============== norm_numerator_reduce_in_master ============================");
        display(norm_numerator_reduce_in_master, pod_context.num_snapshots);
        LOG("=============== norm_denominator_reduce_in_master ============================");
        display(norm_denominator_reduce_in_master, pod_context.num_snapshots);

        allocate(&pod_context.pod_rms_error, pod_context.num_snapshots);

        for (int i = 0; i < pod_context.num_snapshots; i++) {
            pod_context.pod_rms_error[i] = (100 * (float) norm_numerator_reduce_in_master[i]) / (float) norm_denominator_reduce_in_master[i];
        }

        LOG("=============== rms error ============================");
        display(pod_context.pod_rms_error, pod_context.num_snapshots);
    }

    deallocate(&norm_numerator);
    deallocate(&norm_denominator);
    deallocate(&norm_numerator_reduce_in_master);
    deallocate(&norm_denominator_reduce_in_master);
    deallocate(&pod_context.pod_reconstruction_error);
}
