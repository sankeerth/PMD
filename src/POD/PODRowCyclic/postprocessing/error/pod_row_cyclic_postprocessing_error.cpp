#include <math.h>
#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::pod_rms_error_1D_procs_along_row() {
    LOGR("=========== pod_rms_error_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    double *norm_numerator = NULL, *norm_denominator = NULL, *norm_numerator_reduce_global = NULL, *norm_denominator_reduce_global = NULL;

    allocate(&norm_numerator, pod_context.num_snapshots);
    allocate(&norm_denominator, pod_context.num_snapshots);
    allocate(&pod_context.pod_rms_error, pod_context.num_snapshots);

    allocate(&norm_numerator_reduce_global, pod_context.num_snapshots);
    allocate(&norm_denominator_reduce_global, pod_context.num_snapshots);

    initialize(norm_numerator, pod_context.num_snapshots);
    initialize(norm_denominator, pod_context.num_snapshots);

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        unsigned long stride = i * pod_context.num_snapshot_points_per_proc;
        for (unsigned long j = 0; j < pod_context.num_snapshot_points_per_proc; j++) {
            norm_numerator[i] += ((double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j])) * ((double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j]));
            norm_denominator[i] += (double) pod_context.truncated_snapshots[stride + j] * (double) pod_context.truncated_snapshots[stride + j];
        }
    }

    MPI_Allreduce(norm_numerator, norm_numerator_reduce_global, pod_context.num_snapshots, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(norm_denominator, norm_denominator_reduce_global, pod_context.num_snapshots, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    LOG("=============== norm_numerator ============================");
    display(norm_numerator, pod_context.num_snapshots);
    LOG("=============== norm_denominator ============================");
    display(norm_denominator, pod_context.num_snapshots);

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        norm_numerator_reduce_global[i] = sqrt(norm_numerator_reduce_global[i]);
        norm_denominator_reduce_global[i] = sqrt(norm_denominator_reduce_global[i]);

        pod_context.pod_rms_error[i] = (float(100) * (float) norm_numerator_reduce_global[i]) / (float) norm_denominator_reduce_global[i];
    }

    LOG("=============== rms error ============================");
    display(pod_context.pod_rms_error, pod_context.num_snapshots);

    deallocate(&norm_numerator);
    deallocate(&norm_denominator);
    deallocate(&norm_numerator_reduce_global);
    deallocate(&norm_denominator_reduce_global);
}
