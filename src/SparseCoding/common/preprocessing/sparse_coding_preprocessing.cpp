#include <algorithm>
#include "../../../common/headers/log.h"
#include "../../sparse_coding.h"

void SparseCoding::initial_guess_sparse_modes() {
    LOGR("=========== initial_guess_sparse_modes ===========", sparse_context.my_rank, sparse_context.master);

    vector<int> random;

    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        random.push_back(i);
    }

    random_shuffle(random.begin(), random.end());

    LOGD("sparse_context.num_modes_in_my_rank", sparse_context.num_modes_in_my_rank);

    LOG_("Random shuffle of snapshots from which sparse modes are chosen:");
    for (vector<int>::iterator it = random.begin(); it != random.end(); ++it) {
        LOG_(*it);
    }
    NEWLINE;

    allocate(&sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            sparse_context.sparse_modes[i * sparse_context.rank_eigen_values + j] = sparse_context.pod_coefficients[random[i] * sparse_context.rank_eigen_values + j];
        }
    }

    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);
}

void SparseCoding::normalize_sparse_modes() {
    LOGR("=========== normalize_sparse_modes ===========", sparse_context.my_rank, sparse_context.master);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        normalize(&sparse_context.sparse_modes[i * sparse_context.rank_eigen_values], sparse_context.rank_eigen_values);
    }

    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);
}

void SparseCoding::initialize_eplison_convergence_value() {
    LOGR("=========== initialize_eplison_convergence_value ===========", sparse_context.my_rank, sparse_context.master);

    float epsilon = 0;
    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            epsilon += sparse_context.pod_coefficients[i * sparse_context.rank_eigen_values + j] * sparse_context.pod_coefficients[i * sparse_context.rank_eigen_values + j];
        }
    }

    MPI_Allreduce(&epsilon, &sparse_context.epsilon, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    sparse_context.epsilon = 1e-6 * sqrt(sparse_context.epsilon);

    LOGD("sparse_context.epsilon", sparse_context.epsilon);
}
