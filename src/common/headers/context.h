#ifndef CONTEXT_H
#define CONTEXT_H

#include <string>

class Context {
  public:
    int task;
    int job;
    bool is_data_transposed_in_POD_1D_col_cyclic;
    int num_snapshots;
    int num_modes;
    int dimensions;
    int imax, jmax, kmax;
    int it_min, it_max;
    int jt_min, jt_max;
    int kt_min, kt_max;
    int total_num_solution_files;
    int file_interval;
    int start_index_of_snapshots;
    std::string mesh_file;
    std::string path_to_solution_files;
    std::string solution_prefix;
    std::string solution_extension;
    std::string path_to_output_directory;
    bool compute_pod_reconstruction_error;
    bool compute_sparse_coding_reconstruction_error;
    bool is_write_pod_modes_to_text_format;
    bool is_write_pod_modes_to_binary_format;
    bool is_write_coefficients_and_error_to_binary_format;
    float sparsity;
    float epsilon_rank;

    Context(const Context& context)
        : task(context.task)
        , job(context.job)
        , is_data_transposed_in_POD_1D_col_cyclic(context.is_data_transposed_in_POD_1D_col_cyclic)
        , num_snapshots(context.num_snapshots)
        , num_modes(context.num_modes)
        , dimensions(context.dimensions)
        , imax(context.imax)
        , jmax(context.jmax)
        , kmax(context.kmax)
        , it_min(context.it_min)
        , it_max(context.it_max)
        , jt_min(context.jt_min)
        , jt_max(context.jt_max)
        , kt_min(context.kt_min)
        , kt_max(context.kt_max)
        , total_num_solution_files(context.total_num_solution_files)
        , file_interval(context.file_interval)
        , start_index_of_snapshots(context.start_index_of_snapshots)
        , mesh_file(context.mesh_file)
        , path_to_solution_files(context.path_to_solution_files)
        , solution_prefix(context.solution_prefix)
        , solution_extension(context.solution_extension)
        , path_to_output_directory(context.path_to_output_directory)
        , compute_pod_reconstruction_error(context.compute_pod_reconstruction_error)
        , compute_sparse_coding_reconstruction_error(context.compute_sparse_coding_reconstruction_error)
        , is_write_pod_modes_to_text_format(context.is_write_pod_modes_to_text_format)
        , is_write_pod_modes_to_binary_format(context.is_write_pod_modes_to_binary_format)
        , is_write_coefficients_and_error_to_binary_format(context.is_write_coefficients_and_error_to_binary_format)
        , sparsity(context.sparsity)
        , epsilon_rank(context.epsilon_rank)
    { }

    Context() { }
};

#endif // CONTEXT_H
