#ifndef SPARSE_CODING_H
#define SPARSE_CODING_H

#include <iostream>
#include <math.h>
#include "../common/headers/context.h"
#include "../common/headers/mpi_context.h"
#include "../common/headers/utilities.h"

using namespace std;

class SparseContext : public Context, public MPIContext {
  public:
    SparseContext(const Context& context, const MPIContext mpi_context)
        : Context(context)
        , MPIContext(mpi_context)
        , start_snapshot_num(0)
        , num_bases_active(ceil(num_modes * (1 - sparsity)))
        , max_iterations(1000)
        , rank_eigen_values(0)
        , num_modes_in_my_rank(0)
        , replaced_vector_counter(0)
        , epsilon(0)
    {
        nxt = it_max - it_min + 1;
        nyt = jt_max - jt_min + 1;
        nzt = kt_max - kt_min + 1;

        grid_points_in_one_dim = imax * jmax * kmax;
        truncated_grid_points = nxt * nyt * nzt;

        delegate_snapshots_per_process();
        MPI_Exscan(&snapshots_per_rank, &start_snapshot_num, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    int nxt;
    int nyt;
    int nzt;

    // NOTE: changed from int to unsigned long as number of grid points for large
    // dataset was crossing the max value of int and crashing
    unsigned long grid_points_in_one_dim;
    unsigned long truncated_grid_points;

    std::vector<int> index_of_snapshot_filenames;
    std::vector<int> index_of_snapshot_filenames_dummy;

    int rank_eigen_values;
    int num_bases_active;
    float epsilon;
    int max_iterations;
    int snapshots_per_rank;
    int num_modes_in_my_rank;
    int start_snapshot_num;
    int replaced_vector_counter;
    int rows_sparse_modes_transpose_local;
    int cols_sparse_modes_transpose_local;

    float *eigen_values = NULL;
    float *pod_bases = NULL;
    float *pod_coefficients = NULL;
    float *pod_coefficients_dummy = NULL;
    float *pod_coefficients_transpose = NULL;
    float *sparse_modes_transpose = NULL;
    float *sparse_modes = NULL;
    float *coefficient_matrix = NULL;
    float *coefficient_matrix_transpose = NULL;
    float *updated_modes = NULL;
    float *sparse_coding_rms_error = NULL;

    void delegate_snapshots_per_process() {
        int remainder = num_snapshots % num_procs;
        snapshots_per_rank = num_snapshots / num_procs;

        if (my_rank < remainder) {
            snapshots_per_rank++;
        }

        for (int i = 0; i < snapshots_per_rank; i++) {
            index_of_snapshot_filenames.push_back(i * num_procs + my_rank);
            index_of_snapshot_filenames_dummy.push_back((i * num_procs + my_rank + 1) % num_snapshots);
        }

        for (int i = 0; i < index_of_snapshot_filenames.size(); i++) {
            if (index_of_snapshot_filenames[i] < num_modes) {
                num_modes_in_my_rank += 1;
            } else {
                break;
            }
        }
    }
};

class SparseCoding {
  public:
    SparseCoding(const Context& context, const MPIContext& mpi_context);

    /* common */

    // io read
    void read_eigen_values();
    void read_pod_modes();
    void read_pod_coefficients();
    void read_pod_coefficients_dummy();

    // io write
    void write_sparse_modes_binary();
    void write_sparse_coefficients_binary();
    void write_sparse_coding_rms_error_binary();

    // preprocessing
    void initial_guess_sparse_modes();
    void initialize_eplison_convergence_value();
    void normalize_sparse_modes();

    // utilities
    void eigen_power(float *A, int rows, int cols, float eig_epsilon, float *eig_vec_l, float *eig_vec_r, float* eig_val_A);
    void eigen_power_parallel(float *A, int rows_A_local, int cols_A_local, int rows_A_global, int cols_A_global, float eig_epsilon, float *eig_vec_l, float *eig_vec_r, float* eig_val_A, int my_rank);
    void pseudo_inverse(float *A, int& m, int& n);
    void pseudo_inverse(float *A, float **pinvA, int &m, int &n, int &rows_pinvA, int &cols_pinvA);

    // postprocessing error
    void sparse_coding_reconstruction_error(float *truncated_snapshots);

    // postprocessing cleanup
    void cleanup_memory();

    /* Parallel */

    // base
    void generate_sparse_modes_parallel();
    void batch_OMP_parallel();
    void KSVD_parallel();
    void update_modes_KSVD_parallel(int& col);
    void replace_non_active_mode_parallel(int col, int &pos);
    void I_clear_dictionary_parallel();

    /* sparse coding */
    void sparse_coding_preprocessing();
    void compute_sparse_coding();
    void compute_sparse_coding_error(float *truncated_snapshots);
    void write_sparse_coding_output_files();

  private:
    SparseContext sparse_context;
};

#endif // SPARSE_CODING_H
