#include "../../../../common/headers/log.h"
#include "../../../sparse_coding.h"

void SparseCoding::cleanup_memory() {
    LOGR("=========== cleanup_memory ===========", sparse_context.my_rank, sparse_context.master);

    deallocate(&sparse_context.eigen_values);
    deallocate(&sparse_context.pod_bases);
    deallocate(&sparse_context.pod_coefficients);
    deallocate(&sparse_context.sparse_modes);
    deallocate(&sparse_context.coefficient_matrix);
    deallocate(&sparse_context.sparse_coding_rms_error);
}
