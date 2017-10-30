#include "../../../../common/headers/log.h"
#include "../../../sparse_coding.h"

void SparseCoding::read_eigen_values() {
    LOGR("=========== read_eigen_values ===========", sparse_context.my_rank, sparse_context.master);

    size_t result;
    string str;
    str.append(sparse_context.path_to_output_directory);
    str.append("eigen_values_bin.b");
    size_t num_eigen_values = fsize(str.c_str()) / 4;
    FILE *binfile = fopen(str.c_str(), "rb");

    // TODO: relying on the size of eigen values file to get the rank calculated in POD
    // Need to see how to propogate the value found in POD to sparse coding if run independently
    sparse_context.rank_eigen_values = num_eigen_values;

    LOGD("rank of snapshot matrix", sparse_context.rank_eigen_values);

    if (sparse_context.num_modes > sparse_context.rank_eigen_values) {
        LOG("Number of modes can not be larger than the rank of the snapshot matrix");
        LOG("Re run the calculation of sparse coding with number of modes lesser or equal to the rank of the snapshots matrix");
    }

    //TODO: have a check to compute sparse coding upto num of modes or upto rank.
    // for now all computations will be upto the modes
    allocate(&sparse_context.eigen_values, sparse_context.num_modes);
    result = fread(sparse_context.eigen_values, sizeof(float), sparse_context.num_modes, binfile);

    display(sparse_context.eigen_values, sparse_context.num_modes);
}

void SparseCoding::read_pod_modes() {
    LOGR("=========== read_pod_modes ===========", sparse_context.my_rank, sparse_context.master);

    size_t result;
    float * buffer = NULL;
    allocate(&sparse_context.pod_bases, 3 * sparse_context.truncated_grid_points * sparse_context.num_modes_in_my_rank);
    // TODO: optimize to read directly to pod_bases than to buffer and then copy it. This will save memory
    allocate(&buffer, 3 * sparse_context.truncated_grid_points);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        string str;
        str.append(sparse_context.path_to_output_directory);
        str.append("pod_modes_bin-");
        str.append(patch::to_string(sparse_context.index_of_snapshot_filenames[i]));
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "rb");
        size_t num_pod_modes = fsize(str.c_str()) / 4;

        // size of pod_bases has be the same as the truncated snapshots
        if (3 * sparse_context.truncated_grid_points != num_pod_modes) {
            LOG("Number of points in single pod_bases file is not equal to 3 * truncated grid points");
            LOG("There is an error in mostly file creation of pod_bases_bin file.");
            MPI_Finalize();
            exit(0);
        }

        result = fread(buffer, sizeof(float), num_pod_modes, binfile);
        for (unsigned long j = 0; j < num_pod_modes; j++) {
            sparse_context.pod_bases[i * num_pod_modes + j] = buffer[j];
        }

        fclose(binfile);
    }

    deallocate(&buffer);
    display(sparse_context.pod_bases, 3 * sparse_context.truncated_grid_points * sparse_context.num_modes_in_my_rank);
}

void SparseCoding::read_pod_coefficients() {
    LOGR("=========== read_pod_coefficients ===========", sparse_context.my_rank, sparse_context.master);

    size_t result;
    float * buffer = NULL;
    allocate(&sparse_context.pod_coefficients, sparse_context.rank_eigen_values * sparse_context.snapshots_per_rank);
    // TODO: optimize to read directly to pod_bases than to buffer and then copy it. This will save memory
    allocate(&buffer, sparse_context.rank_eigen_values);

    for (int i = 0; i < sparse_context.index_of_snapshot_filenames.size(); i++) {
        string str;
        str.append(sparse_context.path_to_output_directory);
        str.append("pod_coefficients_bin-");
        str.append(patch::to_string(sparse_context.total_num_solution_files + sparse_context.start_index_of_snapshots + (sparse_context.index_of_snapshot_filenames[i] * sparse_context.file_interval)),\
                           1, patch::to_string(sparse_context.total_num_solution_files).length()-1);
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "rb");

        result = fread(buffer, sizeof(float), sparse_context.rank_eigen_values, binfile);
        for (unsigned long j = 0; j < sparse_context.rank_eigen_values; j++) {
            sparse_context.pod_coefficients[i * sparse_context.rank_eigen_values + j] = buffer[j];
        }

        fclose(binfile);
    }

    deallocate(&buffer);
    display(sparse_context.pod_coefficients, sparse_context.rank_eigen_values * sparse_context.snapshots_per_rank);
}
