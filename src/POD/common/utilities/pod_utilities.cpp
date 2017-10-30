#include <fstream>
#include "../../../common/headers/log.h"
#include "../../../common/headers/utilities.h"
#include "../../pod.h"

void POD::create_snapshots_list() {
    LOGR("=========== create_snapshots_list ===========", pod_context.my_rank, pod_context.master);

    create_file_list("snapshots.list", pod_context.start_index_of_snapshots, pod_context.num_snapshots, pod_context.file_interval, pod_context.path_to_solution_files, \
                                 pod_context.solution_prefix, pod_context.total_num_solution_files, pod_context.solution_extension);
}

void POD::create_output_directory() {
    create_directory(pod_context.path_to_output_directory);
}

void POD::distribute_snapshot_filenames() {
    LOGR("=========== distribute_snapshot_filenames ===========", pod_context.my_rank, pod_context.master);

    char *snapshot_filenames = "snapshots.list";
    std::ifstream is(snapshot_filenames, ios::in);
    string line;

    if (is.is_open()) {
        while(std::getline(is, line)) {
            pod_context.filenames_to_distribute.push_back(line);
        }
    }

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        LOG(pod_context.filenames_to_distribute[i]);
    }
}

void POD::make_snapshot_truncated_indices() {
    LOGR("=========== make_snapshot_truncated_indices ===========", pod_context.my_rank, pod_context.master);

    allocate(&pod_context.truncated_snapshot_indices, pod_context.truncated_grid_points);
    int index = 0;

    for (int i = pod_context.kt_min; i <= pod_context.kt_max; i++) {
        for (int j = pod_context.jt_min; j <= pod_context.jt_max; j++) {
            for (int k = pod_context.it_min; k <= pod_context.it_max; k++, index++) {
                pod_context.truncated_snapshot_indices[index] = (pod_context.imax * pod_context.jmax * i) + (pod_context.imax * j) + k;
            }
        }
    }

    LOG("=========== truncated_snapshot_indices ===========");
    display(pod_context.truncated_snapshot_indices, pod_context.truncated_grid_points);
}
