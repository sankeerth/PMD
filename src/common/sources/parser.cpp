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
        file >> context.num_snapshots;
        file >> context.num_modes;
        file >> context.dimensions;
        file >> context.procs_along_row;
        file >> context.imax >> context.jmax >> context.kmax;
        file >> context.it_min >>  context.it_max >>  context.jt_min >>  context.jt_max >>  context.kt_min >>  context.kt_max;
        file >> context.sparsity;
        file >> context.total_num_solution_files;
        file >> context.file_interval;
        file >> context.start_index_of_snapshots;
        file >> context.mesh_file;
        file >> context.path_to_solution_files;
        file >> context.solution_prefix;
        file >> context.solution_extension;
        file >> context.path_to_output_directory;
        file >> context.is_rank_equal_to_num_modes;
        file >> context.compute_pod_reconstruction_error;
        file >> context.compute_sparse_coding_reconstruction_error;
        file >> context.is_write_pod_modes_to_text_format;
        file >> context.is_write_pod_modes_to_binary_format;
        file >> context.is_write_coefficients_and_error_to_binary_format;
        file >> context.epsilon_rank;
    }

    file.close();
}
