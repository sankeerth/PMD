#ifndef POD_H
#define POD_H

#include <vector>
#include "../common/headers/context.h"
#include "../common/headers/mpi_context.h"
#include "../common/headers/utilities.h"

class PODContext : public Context, public MPIContext {
  public:
    PODContext(const Context& context, const MPIContext mpi_context)
        : Context(context)
        , MPIContext(mpi_context)
        , start_snapshot_num(0)
        , rank_eigen_values(0)
        , num_modes_in_my_rank(0)
    {
        nxt = it_max - it_min + 1;
        nyt = jt_max - jt_min + 1;
        nzt = kt_max - kt_min + 1;

        grid_points_in_one_dim = imax * jmax * kmax;
        grid_points_in_all_dim = dimensions * grid_points_in_one_dim;
        truncated_grid_points = nxt * nyt * nzt;
        truncated_grid_points_in_all_dim = dimensions * truncated_grid_points;

        if (task == 0 and job == 2) {
            num_procs_along_row = num_procs;
            num_procs_along_col = 1;
        } else {
            num_procs_along_row = 1;
            num_procs_along_col = num_procs;
        }

        delegate_snapshots_per_process();
        distribute_num_snapshot_points_per_process();
        MPI_Exscan(&snapshots_per_rank, &start_snapshot_num, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    int start_snapshot_num;
    int nxt;
    int nyt;
    int nzt;

    // NOTE: changed from int to unsigned long as number of grid points for large
    // dataset was crossing the max value of int and crashing
    unsigned long grid_points_in_one_dim;
    unsigned long grid_points_in_all_dim;
    unsigned long truncated_grid_points;
    unsigned long truncated_grid_points_in_all_dim;

    int num_plots;

    float *x_grid = NULL;
    float *y_grid = NULL;
    float *z_grid = NULL;

    float *xt_grid = NULL;
    float *yt_grid = NULL;
    float *zt_grid = NULL;

    float *dxc_grid = NULL;
    float *dyc_grid = NULL;
    float *dzc_grid = NULL;

    float *vol_grid = NULL;
    float *truncated_vol_grid = NULL;

    std::vector<std::string> filenames_to_distribute;
    std::vector<int> index_of_snapshot_filenames;

    int *truncated_snapshot_indices = NULL;
    float **snapshots = NULL;
    float *truncated_snapshots = NULL;
    float *truncated_snapshots_transpose = NULL;
    float *average = NULL;
    float *covariance_matrix = NULL;
    float *eigen_values = NULL;
    float *left_singular_vectors = NULL;
    float *pod_bases = NULL;
    float *pod_bases_transpose = NULL;
    float *pod_coefficients = NULL;
    float *pod_reconstruction_error = NULL;
    float *pod_reconstructed_snapshots_transpose = NULL;
    float *pod_rms_error = NULL;

    int snapshots_per_rank;
    int rank_eigen_values;
    int num_modes_in_my_rank;
    unsigned long num_snapshot_points_per_proc;
    unsigned long size_of_eigen_values;
    unsigned long size_of_left_singular_vectors;

    void delegate_snapshots_per_process() {
        int remainder = num_snapshots % num_procs;
        snapshots_per_rank = num_snapshots / num_procs;

        if (my_rank < remainder) {
            snapshots_per_rank++;
        }

        for (int i = 0; i < snapshots_per_rank; i++) {
            index_of_snapshot_filenames.push_back(i * num_procs + my_rank);
        }

        for (int i = 0; i < index_of_snapshot_filenames.size(); i++) {
            if (index_of_snapshot_filenames[i] < num_modes) {
                num_modes_in_my_rank += 1;
            } else {
                break;
            }
        }
    }

    void distribute_num_snapshot_points_per_process() {
        int remainder = truncated_grid_points_in_all_dim % num_procs;
        num_snapshot_points_per_proc = truncated_grid_points_in_all_dim / num_procs;

        if (my_rank < remainder) {
            num_snapshot_points_per_proc++;
        }
    }

    ~PODContext() {

    }

  private:
    PODContext() { }
};

class POD
{
  public:
    POD(const Context& context, const MPIContext& mpi_context);
    /* common */

    // utilites
    void create_snapshots_list();
    void distribute_snapshot_filenames();
    void make_snapshot_truncated_indices();
    void create_output_directory();
    float* get_truncated_snapshots();

    // preprocessing
    void verify_grid(int imax_check, int jmax_check, int kmax_check);
    void read_mesh(bool write_pod_modes);
    void compute_truncated_grid(float *grid, axis ax);
    void make_grid_with_averaging_points(float *grid, axis ax);
    void compute_differential_of_grid(axis ax);
    void grid_metrics();
    void compute_mesh_volume();
    void distribute_truncated_vol_grid();
    void transpose_truncated_snapshots();

    // base
    void compute_covariance_matrix();
    void compute_svd();
    void compute_rank_eigen_values();
    void compute_pod_modes();
    void compute_pod_coefficients();
    void compute_pod_coefficients_from_modes_and_snapshots();

    // io read
    void read_snapshots();

    // io write
    void write_eigen_values_binary();
    void write_pod_rms_error_binary();

    // postprocessing error
    void pod_reconstruction_error();

    // postprocessing cleanup
    void cleanup_memory();
    void deallocate_snapshots_recon_errror();

    /* PODColCyclic */

    // preprocessing
    void compute_average_1D_procs_along_col();
    void compute_fluctuating_component_1D_procs_along_col();
    void compute_snapshot_matrix_1D_procs_along_col();
    void modify_left_singular_matrix_1D_procs_along_col();

    // io read
    void read_truncated_snapshots_1D_procs_along_col();
    void read_mean_flow_binary_1D_procs_along_col();
    void read_pod_modes_1D_procs_along_col();

    // io write
    void write_mean_flow_binary_1D_procs_along_col();
    void write_pod_modes_binary_1D_procs_along_col();
    void write_pod_coefficients_binary_1D_procs_along_col();
    void write_pod_modes_text_format_1D_procs_along_col();

    // postprocessing error
    void pod_rms_error_1D_procs_along_col();
    void pod_rms_error_1D_procs_along_col_data_transposed();

    /* PODRowCyclic */

    // preprocessing
    void compute_average_1D_procs_along_row();
    void compute_fluctuating_component_1D_procs_along_row();
    void compute_snapshot_matrix_1D_procs_along_row();
    void modify_left_singular_matrix_1D_procs_along_row();

    // io read
    void read_truncated_snapshots_1D_procs_along_row();

    // io write
    void write_mean_flow_binary_1D_procs_along_row();
    void write_pod_modes_binary_1D_procs_along_row();
    void write_pod_coefficients_binary_1D_procs_along_row();

    // postprocessing
    void pod_rms_error_1D_procs_along_row();

    /* pod */
    void mesh_processing();
    void snapshots_preprocessing_1D_procs_along_col();
    void snapshots_preprocessing_1D_procs_along_row();
    void compute_pod_1D_procs_along_col();
    void compute_pod_1D_procs_along_row();
    void compute_pod_error_1D_procs_along_col();
    void compute_pod_error_1D_procs_along_row();
    void write_pod_output_files_1D_procs_along_col();
    void write_pod_output_files_1D_procs_along_row();

    double start_time;
  private:
    PODContext pod_context;
};

#endif // POD_H
