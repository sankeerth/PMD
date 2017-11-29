#include "../../../../common/headers/log.h"
#include "../../../../common/headers/scalapack_helpers.h"
#include "../../../sparse_coding.h"

void SparseCoding::write_sparse_modes_binary() {
    LOGR("=========== write_sparse_modes_binary ===========", sparse_context.my_rank, sparse_context.master);

    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("transf_matrix_bin-");
        str.append(patch::to_string(sparse_context.index_of_snapshot_filenames[i]));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < sparse_context.rank_eigen_values; j++) {
            fwrite(&sparse_context.sparse_modes[i * sparse_context.rank_eigen_values + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void SparseCoding::write_sparse_modes_in_original_domain_binary() {
    LOGR("=========== write_sparse_modes_in_original_domain_binary ===========", sparse_context.my_rank, sparse_context.master);
    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("sparse_modes_original_domain_bin-");
        str.append(patch::to_string(sparse_context.index_of_snapshot_filenames[i]));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < sparse_context.truncated_grid_points_in_all_dim; j++) {
            fwrite(&sparse_context.pod_modes_into_sparse_modes[i * sparse_context.truncated_grid_points_in_all_dim + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void SparseCoding::write_corrected_sparse_coefficients_binary() {
    LOGR("=========== write_corrected_sparse_coefficients_binary ===========", sparse_context.my_rank, sparse_context.master);

    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    for (int i = 0, file_num = sparse_context.my_rank; i < sparse_context.index_of_snapshot_filenames.size(); i++, file_num += sparse_context.num_procs) {
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("corrected_sparse_coefficients_bin-");
        str.append(patch::to_string(sparse_context.total_num_solution_files + sparse_context.start_index_of_snapshots + (file_num * sparse_context.file_interval)),\
                   1, patch::to_string(sparse_context.total_num_solution_files).length()-1);
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < sparse_context.num_modes; j++) {
            fwrite(&sparse_context.corrected_sparse_coeff[i * sparse_context.num_modes + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void SparseCoding::write_sparse_coefficients_binary_parallel() {
    LOGR("=========== write_sparse_coefficients_binary ===========", sparse_context.my_rank, sparse_context.master);

    int rows_coefficient_matrix_local, cols_coefficient_matrix_local;

    matrix_transpose(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots, sparse_context.num_modes, &sparse_context.coefficient_matrix, rows_coefficient_matrix_local,\
                     cols_coefficient_matrix_local, sparse_context.num_procs_along_row, sparse_context.num_procs_along_col, 1, 1, 1, 1);

    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    for (int i = 0, file_num = sparse_context.my_rank; i < sparse_context.index_of_snapshot_filenames.size(); i++, file_num += sparse_context.num_procs) {
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("sparse_coefficients_bin-");
        str.append(patch::to_string(sparse_context.total_num_solution_files + sparse_context.start_index_of_snapshots + (file_num * sparse_context.file_interval)),\
                   1, patch::to_string(sparse_context.total_num_solution_files).length()-1);
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < sparse_context.num_modes; j++) {
            fwrite(&sparse_context.coefficient_matrix[i * sparse_context.num_modes + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void SparseCoding::write_sparse_coefficients_binary_serial() {
    LOGR("=========== write_sparse_coefficients_binary_serial ===========", sparse_context.my_rank, sparse_context.master);

    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    for (int i = 0, file_num = sparse_context.my_rank; i < sparse_context.index_of_snapshot_filenames.size(); i++, file_num += sparse_context.num_procs) {
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("sparse_coefficients_bin-");
        str.append(patch::to_string(sparse_context.total_num_solution_files + sparse_context.start_index_of_snapshots + (file_num * sparse_context.file_interval)),\
                   1, patch::to_string(sparse_context.total_num_solution_files).length()-1);
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < sparse_context.num_modes; j++) {
            fwrite(&sparse_context.coefficient_matrix[i * sparse_context.num_modes + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void SparseCoding::write_sparse_coding_rms_error_binary() {
    LOGR("=========== write_sparse_coding_rms_error_binary ===========", sparse_context.my_rank, sparse_context.master);

    // create output directory if not present
    create_directory(sparse_context.path_to_sparse_output_directory);

    int files_to_write = MIN(sparse_context.rank_eigen_values, sparse_context.num_modes);

    for (int i = 0; (i < sparse_context.index_of_snapshot_filenames.size()) && (sparse_context.index_of_snapshot_filenames[i] < files_to_write); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(sparse_context.path_to_sparse_output_directory);
        str.append("sparse_coding_rms_error_bin-");
        str.append(patch::to_string(sparse_context.index_of_snapshot_filenames[i]));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        fwrite(&sparse_context.sparse_coding_rms_error[i], sizeof(float), 1, binfile);

        fclose(binfile);
    }
}
