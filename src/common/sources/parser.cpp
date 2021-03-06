#include <fstream>
#include "../headers/parser.h"

/**
    Parses input files passed as argument to the executable

    @param void
    @return void
*/
void Parser::parse_input_file(char *input_file, Context& context) {
    std::ifstream file(input_file);
    if (file.is_open()) {
        file >> context.task;
        file >> context.job;
        file >> context.is_data_transposed_in_POD_1D_col_cyclic;
        file >> context.num_snapshots;
        file >> context.num_modes;
        file >> context.dimensions;
        file >> context.imax >> context.jmax >> context.kmax;
        file >> context.it_min >>  context.it_max >>  context.jt_min >>  context.jt_max >>  context.kt_min >>  context.kt_max;
        file >> context.total_num_solution_files;
        file >> context.file_interval;
        file >> context.start_index_of_snapshots;
        file >> context.mesh_file;
        file >> context.path_to_solution_files;
        file >> context.solution_prefix;
        file >> context.solution_extension;
        file >> context.path_to_output_directory;
        file >> context.path_to_sparse_output_directory;
        file >> context.compute_pod_reconstruction_error;
        file >> context.compute_sparse_coding_reconstruction_error;
        file >> context.is_write_pod_modes_to_text_format;
        file >> context.is_write_pod_modes_to_binary_format;
        file >> context.is_write_pod_coefficients_and_error_to_binary_format;
        file >> context.is_write_sparse_transformation_matrix;
        file >> context.is_write_sparse_coefficients;
        file >> context.is_write_sparse_modes_in_original_domain;
        file >> context.is_write_corrected_sparse_coefficients;
        file >> context.is_write_sparse_reconstruction_error;
        file >> context.sparsity;
        file >> context.epsilon_rank;
        file >> context.convergence_criteria;
    }

    file.close();
}
