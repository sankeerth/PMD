#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::cleanup_memory() {
    LOGR("=========== cleanup_memory ===========", pod_context.my_rank, pod_context.master);

    deallocate(&pod_context.eigen_values);
    deallocate(&pod_context.pod_bases);
    deallocate(&pod_context.pod_coefficients);
    deallocate(&pod_context.pod_rms_error);
}

void POD::deallocate_snapshots_recon_errror() {
    LOGR("=========== deallocate_snapshots_recon_errror ===========", pod_context.my_rank, pod_context.master);

    deallocate(&pod_context.truncated_snapshots);
    deallocate(&pod_context.pod_reconstruction_error);
}
