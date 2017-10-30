#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../pod.h"

void POD::write_pod_modes_binary_1D_procs_along_row() {
    LOGR("=========== write_pod_modes_binary_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    float *pod_bases_transpose = NULL;
    int rows_local_pod_bases_transpose, cols_local_pod_bases_transpose;

    matrix_transpose(pod_context.pod_bases, pod_context.truncated_grid_points_in_all_dim, pod_context.num_snapshots, &pod_bases_transpose, rows_local_pod_bases_transpose,\
                     cols_local_pod_bases_transpose, pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1);

    LOG("=============== pod_coefficients_transpose ============================");
    display(pod_bases_transpose, rows_local_pod_bases_transpose * cols_local_pod_bases_transpose);

    // create output directory if not present
    create_pod_modes_directory();
    int files_to_write = MIN(pod_context.rank_eigen_values, pod_context.num_modes);

    for (int i = 0; (i < pod_context.index_of_snapshot_filenames.size()) && (pod_context.index_of_snapshot_filenames[i] < files_to_write); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_modes_bin-");
        str.append(patch::to_string(pod_context.index_of_snapshot_filenames[i]));
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            fwrite(&pod_bases_transpose[j * pod_context.snapshots_per_rank + i], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }

    deallocate(&pod_bases_transpose);
}

void POD::write_pod_coefficients_binary_1D_procs_along_row() {
    LOGR("=========== write_pod_coefficients_binary_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    float *pod_coefficients_transpose = NULL;
    int rows_local_pod_coefficients_transpose, cols_local_pod_coefficients_transpose;

    matrix_transpose(pod_context.pod_coefficients, pod_context.num_snapshots, pod_context.num_snapshots, &pod_coefficients_transpose, rows_local_pod_coefficients_transpose,\
                     cols_local_pod_coefficients_transpose, pod_context.num_procs_along_row, pod_context.num_procs_along_col, 1, 1, 1, 1);

    LOG("=============== pod_coefficients_transpose ============================");
    display(pod_coefficients_transpose, rows_local_pod_coefficients_transpose * cols_local_pod_coefficients_transpose);

    for (int i = 0, file_num = pod_context.my_rank; i < pod_context.snapshots_per_rank; i++, file_num += pod_context.num_procs) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_coefficients_bin-");
        str.append(patch::to_string(pod_context.total_num_solution_files + pod_context.start_index_of_snapshots + (file_num * pod_context.file_interval)),\
                   1, patch::to_string(pod_context.total_num_solution_files).length()-1);
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < pod_context.rank_eigen_values; j++) {
            fwrite(&pod_coefficients_transpose[j * pod_context.snapshots_per_rank + i], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }

    deallocate(&pod_coefficients_transpose);
}

void POD::write_pod_rms_error_binary_1D_procs_along_row() {
    LOGR("=========== write_pod_rms_error_binary_1D_procs_along_row ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_pod_modes_directory();

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_rms_error_bin-");
        str.append(patch::to_string(i));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        fwrite(&pod_context.pod_rms_error[i], sizeof(float), 1, binfile);

        fclose(binfile);
    }
}
