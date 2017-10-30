#include <math.h>
#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::pod_rms_error_1D_procs_along_col() {
    LOGR("=========== compute_average_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    double *norm_numerator = NULL, *norm_denominator = NULL;

    allocate(&norm_numerator, pod_context.snapshots_per_rank);
    allocate(&norm_denominator, pod_context.snapshots_per_rank);
    allocate(&pod_context.pod_rms_error, pod_context.snapshots_per_rank);

    initialize(norm_numerator, pod_context.snapshots_per_rank);
    initialize(norm_denominator, pod_context.snapshots_per_rank);

    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        unsigned long stride = i * pod_context.truncated_grid_points_in_all_dim;
        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            norm_numerator[i] += (double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j]) * (double) (pod_context.truncated_snapshots[stride + j] - pod_context.pod_reconstruction_error[stride + j]);
            norm_denominator[i] += (double) (pod_context.truncated_snapshots[stride + j] * pod_context.truncated_snapshots[stride + j]);
        }

        norm_numerator[i] = sqrt(norm_numerator[i]);
        norm_denominator[i] = sqrt(norm_denominator[i]);

        pod_context.pod_rms_error[i] = (float(100) * (float) norm_numerator[i]) / (float) norm_denominator[i];
    }

    LOG("=============== norm_numerator ============================");
    display(norm_numerator, pod_context.snapshots_per_rank);
    LOG("=============== norm_denominator ============================");
    display(norm_denominator, pod_context.snapshots_per_rank);

    LOG("=============== rms error ============================");
    display(pod_context.pod_rms_error, pod_context.snapshots_per_rank);

    deallocate(&norm_numerator);
    deallocate(&norm_denominator);
}
