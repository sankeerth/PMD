#include "../../../common/headers/log.h"
#include "../../../common/headers/scalapack_helpers.h"
#include "../../sparse_coding.h"

void SparseCoding::generate_sparse_modes_parallel() {
    LOGR("=========== generate_sparse_modes_parallel ===========", sparse_context.my_rank, sparse_context.master);

    LOGDR("sparsity", sparse_context.sparsity, sparse_context.my_rank, sparse_context.master);

    int rows_pod_coefficients_tranpose_local, cols_pod_coefficients_tranpose_local, iteration = 0, rows_errors_local, cols_errors_local;
    float epsilon_val = 1e10, error_local, error_global, prev_error;
    float *errors = NULL;

    matrix_transpose(sparse_context.pod_coefficients, sparse_context.rank_eigen_values, sparse_context.num_snapshots, &sparse_context.pod_coefficients_transpose, rows_pod_coefficients_tranpose_local,\
                     cols_pod_coefficients_tranpose_local, 1, sparse_context.num_procs, 1, 1, 1, 1);

    LOG("=============== pod_coefficients_transpose ============================");
    display(sparse_context.pod_coefficients_transpose, rows_pod_coefficients_tranpose_local * cols_pod_coefficients_tranpose_local);

    allocate(&sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.snapshots_per_rank);

    while (epsilon_val > sparse_context.epsilon && iteration < sparse_context.max_iterations) {

        matrix_transpose(sparse_context.sparse_modes, sparse_context.rank_eigen_values, sparse_context.num_modes, &sparse_context.sparse_modes_transpose, sparse_context.rows_sparse_modes_transpose_local,\
                         sparse_context.cols_sparse_modes_transpose_local, 1, sparse_context.num_procs, sparse_context.rank_eigen_values, 1, sparse_context.num_modes, 1);

        LOG("=============== sparse_modes_transpose ============================");
        display(sparse_context.sparse_modes_transpose, sparse_context.rows_sparse_modes_transpose_local * sparse_context.cols_sparse_modes_transpose_local);

        batch_OMP_parallel();
        MPI_Barrier(MPI_COMM_WORLD);
        KSVD_parallel();
        MPI_Barrier(MPI_COMM_WORLD);
        I_clear_dictionary_parallel();
        MPI_Barrier(MPI_COMM_WORLD);

        LOGR("=============== sparse_modes ============================", sparse_context.my_rank, sparse_context.master);
        display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);

        matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.coefficient_matrix_transpose, &errors, rows_errors_local, cols_errors_local, sparse_context.rank_eigen_values, sparse_context.num_snapshots,\
                                   sparse_context.num_modes, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

        error_local = 0;
        for (int i = 0; i < cols_errors_local; i++) {
            for (int j = 0; j < rows_errors_local; j++) {
                errors[i * rows_errors_local + j] = (sparse_context.pod_coefficients[i * rows_errors_local + j] - errors[i * rows_errors_local + j]);
                errors[i * rows_errors_local + j] = errors[i * rows_errors_local + j] * errors[i * rows_errors_local + j];
                error_local += errors[i * rows_errors_local + j];
            }
        }

        LOG("=============== errors ============================");
        display(errors, rows_errors_local * cols_errors_local);

        MPI_Allreduce(&error_local, &error_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        error_global = sqrt(error_global/ (sparse_context.num_snapshots * sparse_context.rank_eigen_values));

        if (iteration > 5) {
            epsilon_val = fabs(prev_error - error_global);
        }

        LOGDR_("prev error", prev_error, sparse_context.my_rank, sparse_context.master);
        LOGDR_("error", error_global, sparse_context.my_rank, sparse_context.master);
        LOGDR("epsilon val", epsilon_val, sparse_context.my_rank, sparse_context.master);

        prev_error = error_global;
        iteration += 1;

	 LOGDR("iteration", iteration, sparse_context.my_rank, sparse_context.master);
    }

    LOGD("Total iterations", iteration);

    deallocate(&sparse_context.sparse_modes_transpose);
    deallocate(&sparse_context.pod_coefficients_transpose);

    // deallocating right after use to reduce peak mem usage
    if(!(sparse_context.is_write_sparse_transformation_matrix | sparse_context.is_write_sparse_modes_in_original_domain | sparse_context.is_write_corrected_sparse_coefficients | sparse_context.is_write_sparse_reconstruction_error))
        deallocate((&sparse_context.sparse_modes));

    // deallocating right after use to reduce peak mem usage
    if(!sparse_context.is_write_sparse_coefficients)
        deallocate(&sparse_context.coefficient_matrix);


}

void SparseCoding::batch_OMP_parallel() {
    LOGR("=========== batch_OMP_parallel ===========", sparse_context.my_rank, sparse_context.master);

    float alpha = 1.0, beta = 0.0, max;
    float *G_local = NULL, *G = NULL, *G_temp_buffer = NULL, *x = NULL, *D_prod_x = NULL, *y = NULL, *y_dup = NULL, *y_ = NULL, *PinvG = NULL;
    float *PinvG_prod_y_ = NULL, *G_ = NULL, *G_prod_PinvG_prod_y_ = NULL, *D_ = NULL, *PinvD_ = NULL, *D_tranpose = NULL, *A_ = NULL;
    int *index = NULL, *remote_index = NULL, *rows_and_cols_G = NULL;
    int incx = 1, incy = 1, pos, num_snpashots_in_master, length_of_A_;
    int rows_G_local, cols_G_local, rows_alpha_local, cols_alpha_local, rows_pinv_D_, cols_pinv_D_, rows_D_transpose, cols_D_transpose;

    SCALAPACK_LOG("=============== pod_coefficients ============================");
    display(sparse_context.pod_coefficients, sparse_context.rank_eigen_values * sparse_context.snapshots_per_rank);
    SCALAPACK_LOG("=============== sparse_modes ============================");
    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);

    matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.sparse_modes, &G_local, rows_G_local, cols_G_local, sparse_context.rank_eigen_values, sparse_context.num_modes, sparse_context.num_modes,\
                               'T', 'N', 1, sparse_context.num_procs, sparse_context.rank_eigen_values, 1, sparse_context.rank_eigen_values, 1, true);

    allocate(&G, sparse_context.num_modes * sparse_context.num_modes);
    allocate(&rows_and_cols_G, 2);

    for (int i = 0; i < sparse_context.num_procs; i++) {
        if (sparse_context.my_rank == i) {
            rows_and_cols_G[0] = rows_G_local;
            rows_and_cols_G[1] = cols_G_local;
        }

        MPI_Bcast(rows_and_cols_G, 2, MPI_INT, i, MPI_COMM_WORLD);
        allocate(&G_temp_buffer, rows_and_cols_G[0] * rows_and_cols_G[1]);

        if (sparse_context.my_rank == i) {
            for (int j = 0; j < rows_G_local * cols_G_local; j++) {
                G_temp_buffer[j] = G_local[j];
            }
        }

        MPI_Bcast(G_temp_buffer, rows_and_cols_G[0] * rows_and_cols_G[1], MPI_FLOAT, i, MPI_COMM_WORLD);

        for (int j = 0, index = 0; j < rows_and_cols_G[1]; j++) {
            for (int k = 0; k < rows_and_cols_G[0]; k++, index++) {
                G[(i + j * sparse_context.num_procs) * sparse_context.num_modes + k] = G_temp_buffer[index];
            }
        }

        deallocate(&G_temp_buffer);
    }

    SCALAPACK_LOG("=============== G ============================");
    display(G, sparse_context.num_modes * sparse_context.num_modes);

    matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.pod_coefficients, &D_prod_x, rows_alpha_local, cols_alpha_local, sparse_context.rank_eigen_values, sparse_context.num_modes, sparse_context.num_snapshots,\
                               'T', 'N', 1, sparse_context.num_procs, 1 ,1 ,1, 1, false);

    SCALAPACK_LOG("=============== D_prod_x ============================");
    display(D_prod_x, rows_alpha_local * cols_alpha_local);

    allocate(&y, sparse_context.num_modes);
    allocate(&y_dup, sparse_context.num_modes);
    allocate(&index, sparse_context.num_bases_active);
    allocate(&remote_index, sparse_context.num_bases_active);
    allocate(&D_, sparse_context.num_bases_active * sparse_context.cols_sparse_modes_transpose_local);
    allocate(&x, sparse_context.rank_eigen_values);

    initialize(sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.snapshots_per_rank);

    if (sparse_context.my_rank == sparse_context.master) {
        num_snpashots_in_master = sparse_context.snapshots_per_rank;
    }

    MPI_Bcast(&num_snpashots_in_master, 1, MPI_INT, sparse_context.master, MPI_COMM_WORLD);
    SCALAPACK_LOGD("num_snpashots_in_master", num_snpashots_in_master);

    for (int i = 0; i < num_snpashots_in_master; i++) {
        if (i < sparse_context.snapshots_per_rank) {
            for (int j = 0; j < sparse_context.num_modes; j++) {
                y[j] = D_prod_x[i * sparse_context.num_modes + j];
                y_dup[j] = D_prod_x[i * sparse_context.num_modes + j];
            }

            SCALAPACK_LOG("=============== y ============================");
            display(y, sparse_context.num_modes);

            initialize(index, sparse_context.num_bases_active);

            for (int j = 0, m = j+1; j < sparse_context.num_bases_active; j++, m++) {
                get_abs_max_and_pos(y, sparse_context.num_modes, &max, &pos);
                SCALAPACK_LOGD_("rank", sparse_context.my_rank);
                SCALAPACK_LOGD_("max val", max);
                SCALAPACK_LOGD("pos", pos);
                index[j] = pos;

                SCALAPACK_LOG("=============== index ============================");
                display(index, sparse_context.num_bases_active);

                allocate(&PinvG, m * m);
                allocate(&y_, m);
                allocate(&PinvG_prod_y_, m);
                allocate(&G_, sparse_context.num_modes * m);
                allocate(&G_prod_PinvG_prod_y_, sparse_context.num_modes);

                // truncate y_dup[index[0:m]]
                for (int a = 0; a < m; a++) {
                    y_[a] = y_dup[index[a]];
                }

                // truncated G[index[0:m]][index[0:m]]
                for (int a = 0, c = 0; a < m; a++) {
                    for (int b = 0; b < m; b++, c++) {
                        PinvG[c] = G[index[a] * sparse_context.num_modes + index[b]];
                    }
                }

                pseudo_inverse(PinvG, m, m);

                SCALAPACK_LOG("=============== PinvG ============================");
                display(PinvG, m * m);

                SCALAPACK_LOG("=============== PinvG * y_ ============================");
                sgemv("N", &m, &m, &alpha, PinvG, &m, y_, &incx, &beta, PinvG_prod_y_, &incy);
                display(PinvG_prod_y_, m);

                // truncated G[0:m)[index[0:m]]
                for (int a = 0, c = 0; a < m; a++) {
                    for (int b = 0; b < sparse_context.num_modes; b++, c++) {
                        G_[c] = G[index[a] * sparse_context.num_modes + b];
                    }
                }

                SCALAPACK_LOG("=============== G_ ============================");
                display(G_, sparse_context.num_modes * m);

                SCALAPACK_LOG("=============== G_ * PinvG * y_ ============================");
                sgemv("N", &sparse_context.num_modes, &m, &alpha, G_, &sparse_context.num_modes, PinvG_prod_y_, &incx, &beta, G_prod_PinvG_prod_y_, &incy);
                display(G_prod_PinvG_prod_y_, sparse_context.num_modes);

                SCALAPACK_LOG("=============== updating y ============================");
                for (int a = 0; a < sparse_context.num_modes; a++) {
                    y[a] = y_dup[a] - G_prod_PinvG_prod_y_[a];
                }
                display(y, sparse_context.num_modes);

                SCALAPACK_LOG("=============== readjust y ============================");
                for (int a = 0; a < m; a++) {
                    y[index[a]] = 0;
                }

                deallocate(&PinvG);
                deallocate(&y_);
                deallocate(&PinvG_prod_y_);
                deallocate(&G_);
                deallocate(&G_prod_PinvG_prod_y_);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int j = 0; j < sparse_context.num_procs; j++) {
            if (j == sparse_context.my_rank) {
                duplicate(index, remote_index, sparse_context.num_bases_active);
            }

            MPI_Bcast(remote_index, sparse_context.num_bases_active, MPI_INT, j, MPI_COMM_WORLD);

            for (int k = 0; k < sparse_context.cols_sparse_modes_transpose_local; k++) {
                for (int l = 0; l < sparse_context.num_bases_active; l++) {
                    D_[k * sparse_context.num_bases_active + l] = sparse_context.sparse_modes_transpose[k * sparse_context.rows_sparse_modes_transpose_local + remote_index[l]];
                }
            }

            SCALAPACK_LOG("=============== D_ ============================");
            display(D_, sparse_context.num_bases_active * sparse_context.cols_sparse_modes_transpose_local);

            matrix_transpose(D_, sparse_context.num_bases_active, sparse_context.rank_eigen_values, &D_tranpose, rows_D_transpose, cols_D_transpose, 1, sparse_context.num_procs, sparse_context.num_bases_active, 1, sparse_context.rank_eigen_values, 1);

            SCALAPACK_LOG("=============== D_tranpose ============================");
            display(D_tranpose, rows_D_transpose * cols_D_transpose);

            // TODO: get pseudo_inverse parallel work for rows < cols
            // D_tranpose would not be necessary then
            pseudo_inverse(D_tranpose, &PinvD_, sparse_context.rank_eigen_values, sparse_context.num_bases_active, rows_pinv_D_, cols_pinv_D_);

            SCALAPACK_LOG("=============== PinvD_ ============================");
            display(PinvD_, rows_pinv_D_ * cols_pinv_D_);

            if (j == sparse_context.my_rank && i < sparse_context.snapshots_per_rank) {
                for (int k = 0; k < sparse_context.rank_eigen_values; k++) {
                    x[k] = sparse_context.pod_coefficients[i * sparse_context.rank_eigen_values + k];
                }
                SCALAPACK_LOG("=============== X ============================");
                display(x, sparse_context.rank_eigen_values);
            }

            matrix_vector_mul(PinvD_, sparse_context.num_bases_active, sparse_context.rank_eigen_values, x, &A_, length_of_A_, 'N', 1, 1, 1, sparse_context.num_procs, 1, 1, 1, (j+1), 1, (j+1), 1, 1);

            if (j == sparse_context.my_rank && i < sparse_context.snapshots_per_rank) {
                SCALAPACK_LOG("=============== A_ ============================");
                display(A_, length_of_A_);

                for (int k = 0; k < sparse_context.num_bases_active; k++) {
                    sparse_context.coefficient_matrix[i * sparse_context.num_modes + remote_index[k]] = A_[k];
                }
            }

            deallocate(&D_tranpose);
            deallocate(&PinvD_);
            deallocate(&A_);
        }
    }

    SCALAPACK_LOG("=============== coefficient_matrix ============================");
    display(sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.snapshots_per_rank);

    deallocate(&y);
    deallocate(&y_dup);
    deallocate(&index);
    deallocate(&remote_index);
    deallocate(&rows_and_cols_G);
    deallocate(&D_);
    deallocate(&x);
    deallocate(&G_local);
    deallocate(&G);
    deallocate(&D_prod_x);
}

void SparseCoding::KSVD_parallel() {
    LOGR("=========== KSVD_parallel ===========", sparse_context.my_rank, sparse_context.master);

    int rows_coefficient_matrix_tranpose_local, cols_coefficient_matrix_tranpose_local;
    allocate(&sparse_context.updated_modes, sparse_context.rank_eigen_values);

    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        float max; int pos;
        get_max_and_pos(&sparse_context.coefficient_matrix[i * sparse_context.num_modes], sparse_context.num_modes, &max, &pos);

        round_off_below_diff_max_and_threshold_to_zero(&sparse_context.coefficient_matrix[i * sparse_context.num_modes], sparse_context.num_modes,\
                                                        max, 1e-6);
    }

    matrix_transpose(sparse_context.coefficient_matrix, sparse_context.num_modes, sparse_context.num_snapshots, &sparse_context.coefficient_matrix_transpose, rows_coefficient_matrix_tranpose_local,\
                     cols_coefficient_matrix_tranpose_local, 1, sparse_context.num_procs, sparse_context.num_modes, 1, sparse_context.num_snapshots, 1);

    LOGR("=============== coefficient matrix transpose ============================", sparse_context.my_rank, sparse_context.master);
    display(sparse_context.coefficient_matrix_transpose, rows_coefficient_matrix_tranpose_local * cols_coefficient_matrix_tranpose_local);

    for (int i = 0; i < sparse_context.num_modes; i++) {
        update_modes_KSVD_parallel(i);

        if ((i % sparse_context.num_procs) == sparse_context.my_rank) {
            LOG("=============== updated_modes ============================");
            display(sparse_context.updated_modes, sparse_context.rank_eigen_values);

            int local_column = i / sparse_context.num_procs;

            for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
                sparse_context.sparse_modes[local_column * sparse_context.rank_eigen_values + j] = sparse_context.updated_modes[j];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    LOGR("=============== coefficient matrix transpose ============================", sparse_context.my_rank, sparse_context.master);
    display(sparse_context.coefficient_matrix_transpose, rows_coefficient_matrix_tranpose_local * cols_coefficient_matrix_tranpose_local);

    LOGR("=============== sparse_modes ============================", sparse_context.my_rank, sparse_context.master);
    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);

    //TODO: May be the size of coeff_mat_transpose, updated_modes and sparse_modes are same throughout even if the value is not.
    // Instead of freeing it, probably can be just reinitialized to zero
    deallocate(&sparse_context.updated_modes);
}

void SparseCoding::update_modes_KSVD_parallel(int &col) {
    LOGR("=========== update_modes_KSVD_parallel ===========", sparse_context.my_rank, sparse_context.master);

    float singular_value = 0, *beta_vector = NULL, *coefficient_matrix_relevant_indices = NULL, *errors = NULL;
    int *relevant_data_indices = NULL;
    float epsilon = 1e-10;
    int size, rows_errors_local, cols_errors_local;

    if ((col % sparse_context.num_procs) == sparse_context.my_rank) {
        vector<int> indices = get_indices_nonzero_each_col(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots, col / sparse_context.num_procs);

        LOG("=============== coefficient_matrix_transpose ============================");
        display(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots * sparse_context.num_modes_in_my_rank);

        size = indices.size();
        LOGD("relevant indices size", indices.size());

        allocate(&relevant_data_indices, size);
        for (int i = 0; i < size; i++) {
            relevant_data_indices[i] = indices[i];
        }

        allocate(&beta_vector, size);
    }

    MPI_Bcast(&size, 1, MPI_INT, (col % sparse_context.num_procs), MPI_COMM_WORLD);

    if ((col % sparse_context.num_procs) != sparse_context.my_rank) {
        allocate(&relevant_data_indices, size);
    }

    MPI_Bcast(relevant_data_indices, size, MPI_INT, (col % sparse_context.num_procs), MPI_COMM_WORLD);

    LOGD_("rank", sparse_context.my_rank);
    LOGD("size", size);

    LOG("=============== relevant_data_indices ============================");
    display(relevant_data_indices, size);

    if (size < 1) {
        int pos;
        replace_non_active_mode_parallel((col), pos);
        return;
    }

    allocate(&coefficient_matrix_relevant_indices, size * sparse_context.num_modes_in_my_rank);

    for (int i = 0; i < sparse_context.num_modes_in_my_rank; i++) {
        for (int j = 0; j < size; j++) {
            coefficient_matrix_relevant_indices[i * size + j] = sparse_context.coefficient_matrix_transpose[i * sparse_context.num_snapshots + relevant_data_indices[j]];
        }
    }

    LOG("=============== coefficient_matrix_relevant_indices ============================");
    display(coefficient_matrix_relevant_indices, size * sparse_context.num_modes_in_my_rank);

    matrix_mul_1D_process_grid(coefficient_matrix_relevant_indices, sparse_context.sparse_modes, &errors, rows_errors_local, cols_errors_local, size,\
                               sparse_context.rank_eigen_values, sparse_context.num_modes, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    LOG("=============== errors ============================");
    display(errors, rows_errors_local * cols_errors_local);

    for (int i = 0; i < cols_errors_local; i++) {
        for (int j = 0; j < rows_errors_local; j++) {
            errors[i * rows_errors_local + j] = sparse_context.pod_coefficients_transpose[i * sparse_context.num_snapshots + relevant_data_indices[j]] - errors[i * rows_errors_local + j];
        }
    }

    round_off_below_machine_precision_to_zero(errors, rows_errors_local * cols_errors_local);

    LOG("=============== errors ============================");
    display(errors, rows_errors_local * cols_errors_local);

    eigen_power_parallel(errors, rows_errors_local, cols_errors_local, size, sparse_context.rank_eigen_values, epsilon, sparse_context.updated_modes, beta_vector, &singular_value, (col % sparse_context.num_procs));

    if ((col % sparse_context.num_procs) == sparse_context.my_rank) {
        int local_column = col / sparse_context.num_procs;

        for (int i = 0; i < size; i++) {
            sparse_context.coefficient_matrix_transpose[local_column * sparse_context.num_snapshots + relevant_data_indices[i]] = singular_value * beta_vector[i];
        }
    }

    LOG("=============== coefficient_matrix_transpose ============================");
    display(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots * sparse_context.num_modes_in_my_rank);

    deallocate(&beta_vector);
    deallocate(&relevant_data_indices);
    deallocate(&coefficient_matrix_relevant_indices);
    deallocate(&errors);
}

void SparseCoding::replace_non_active_mode_parallel(int col, int &pos) {
    LOGR("=========== replace_non_active_mode_parallel ===========", sparse_context.my_rank, sparse_context.master);

    float *error_norm_vec = NULL, *errors = NULL, *max_error_norm_vec = NULL, max, max_proc_pos, norm_updated_modes;
    int rows_errors_local, cols_errors_local;

    allocate(&error_norm_vec, sparse_context.snapshots_per_rank);
    allocate(&max_error_norm_vec, sparse_context.num_procs);

    matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.coefficient_matrix_transpose, &errors, rows_errors_local, cols_errors_local, sparse_context.rank_eigen_values, sparse_context.num_snapshots,\
                               sparse_context.num_modes, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    // TODO: same optimization in serial code
    for (int i = 0; i < cols_errors_local; i++) {
        float error = 0;
        for (int j = 0; j < rows_errors_local; j++) {
            errors[i * rows_errors_local + j] = sparse_context.pod_coefficients[i * rows_errors_local + j] - errors[i * rows_errors_local + j];
            errors[i * rows_errors_local + j] = errors[i * rows_errors_local + j] * errors[i * rows_errors_local + j];
            error += errors[i * rows_errors_local + j];
        }
        error_norm_vec[i] = error;
    }

    LOG("=============== error_norm_vec ============================");
    display(error_norm_vec, cols_errors_local);

    pos = 0;
    max = error_norm_vec[0];
    for (int i = 0; i < sparse_context.snapshots_per_rank; i++) {
        if (error_norm_vec[i] > max) {
            pos = i;
            max = error_norm_vec[i];
        }
    }

    MPI_Allgather(&max, 1, MPI_FLOAT, max_error_norm_vec, 1, MPI_FLOAT, MPI_COMM_WORLD);

    max_proc_pos = 0;
    max = max_error_norm_vec[0];
    for (int i = 0; i < sparse_context.num_procs; i++) {
        if (max_error_norm_vec[i] > max) {
            max_proc_pos = i;
            max = max_error_norm_vec[i];
        }
    }

    if (sparse_context.my_rank == max_proc_pos) {
        for (int i = 0; i < sparse_context.rank_eigen_values; i++) {
            sparse_context.updated_modes[i] = sparse_context.pod_coefficients[pos * sparse_context.rank_eigen_values + i];
        }

        norm_updated_modes = norm(sparse_context.updated_modes, sparse_context.rank_eigen_values);

        sparse_context.coefficient_matrix_transpose[(col / sparse_context.num_procs) * sparse_context.num_snapshots + pos] = norm_updated_modes;

        for (int i = 0; i < sparse_context.rank_eigen_values; i++) {
            sparse_context.updated_modes[i] = sparse_context.updated_modes[i] / norm_updated_modes;
        }

        if (max_proc_pos != (col % sparse_context.num_procs)) {
            MPI_Send(sparse_context.updated_modes, sparse_context.rank_eigen_values, MPI_FLOAT, (col % sparse_context.num_procs), 0, MPI_COMM_WORLD);
        }
    }

    if (sparse_context.my_rank == (col % sparse_context.num_procs) && (max_proc_pos != (col % sparse_context.num_procs))) {
        MPI_Recv(sparse_context.updated_modes, sparse_context.rank_eigen_values, MPI_FLOAT, max_proc_pos, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    deallocate(&errors);
    deallocate(&error_norm_vec);
}

void SparseCoding::I_clear_dictionary_parallel() {
    LOGR("=========== I_clear_dictionary_parallel ===========", sparse_context.my_rank, sparse_context.master);

    float T2 = 0.99, T1 = 3.0;
    float *error_norm_vec = NULL, *error_norm_vec_transpose = NULL, *errors = NULL, *G = NULL, *replacement_vector = NULL, *condition = NULL, *cleaned_up_sparse_modes = NULL;
    int position, rows_error_norm_vec_transpose_local, cols_error_norm_vec_transpose_local, rows_errors_local, cols_errors_local, rows_G_local, cols_G_local, condition_criteria = 2;

    allocate(&condition, condition_criteria);
    allocate(&error_norm_vec, sparse_context.snapshots_per_rank);
    allocate(&cleaned_up_sparse_modes, sparse_context.rank_eigen_values);

    LOG("=============== sparse_modes ============================");
    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes_in_my_rank);

    LOG("=============== coefficient_matrix_transpose ============================");
    display(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots * sparse_context.num_modes_in_my_rank);

    matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.coefficient_matrix_transpose, &errors, rows_errors_local, cols_errors_local, sparse_context.rank_eigen_values, sparse_context.num_snapshots,\
                               sparse_context.num_modes, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, false);

    // TODO: same optimization in serial code
    for (int i = 0; i < cols_errors_local; i++) {
        float error = 0;
        for (int j = 0; j < rows_errors_local; j++) {
            errors[i * rows_errors_local + j] = sparse_context.pod_coefficients[i * rows_errors_local + j] - errors[i * rows_errors_local + j];
            errors[i * rows_errors_local + j] = errors[i * rows_errors_local + j] * errors[i * rows_errors_local + j];
            error += errors[i * rows_errors_local + j];
        }
        error_norm_vec[i] = error;
    }

    LOG("=============== errors ============================");
    display(errors, rows_errors_local * cols_errors_local);

    matrix_transpose(error_norm_vec, 1, sparse_context.num_snapshots, &error_norm_vec_transpose, rows_error_norm_vec_transpose_local, cols_error_norm_vec_transpose_local, 1, sparse_context.num_procs, 1, 1, 1, 1);

    LOG("=============== error_norm_vec_transpose ============================");
    display(error_norm_vec_transpose, rows_error_norm_vec_transpose_local * cols_error_norm_vec_transpose_local);

    matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.sparse_modes, &G, rows_G_local, cols_G_local, sparse_context.rank_eigen_values, sparse_context.num_modes, sparse_context.num_modes,\
                               'T', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, true);

    for (int i = 0; i < sparse_context.num_modes; i++) {
        if ((i % sparse_context.num_procs) == sparse_context.my_rank) {
            float max_val;
            int num_of_vals_greater_than_threshold;

            G[((i / sparse_context.num_procs) * sparse_context.num_modes) + i] = 0;

            LOG("=============== G ============================");
            display(G, rows_G_local * cols_G_local);

            get_max_and_pos(&G[((i / sparse_context.num_procs) * sparse_context.num_modes)], sparse_context.num_modes, &max_val, &position);

            num_of_vals_greater_than_threshold = get_num_elements_greater_than_threshold_along_col(sparse_context.coefficient_matrix_transpose, sparse_context.num_snapshots,\
                                                                                                   (i / sparse_context.num_procs), 1e-10);

            condition[0] = max_val;
            condition[1] = num_of_vals_greater_than_threshold;
        }

        MPI_Bcast(condition, condition_criteria, MPI_FLOAT, (i % sparse_context.num_procs), MPI_COMM_WORLD);

        if (condition[0] > T2 || condition[1] <= T1) {
            float max_val;
            int position;

            if (sparse_context.my_rank == sparse_context.master) {
                get_max_and_pos(error_norm_vec_transpose, rows_error_norm_vec_transpose_local * cols_error_norm_vec_transpose_local, &max_val, &position);
                error_norm_vec_transpose[position] = 0;

                LOGD_("position", position);
                LOGD("max value", max_val);
            }

            MPI_Bcast(&position, 1, MPI_INT, sparse_context.master, MPI_COMM_WORLD);

            if ((position % sparse_context.num_procs) == sparse_context.my_rank) {
                int col = position / sparse_context.num_procs;

                for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
                    cleaned_up_sparse_modes[j] = sparse_context.pod_coefficients[col * sparse_context.rank_eigen_values + j];
                }

                normalize(cleaned_up_sparse_modes, sparse_context.rank_eigen_values);

                LOG("=============== cleaned_up_sparse_modes ============================");
                display(cleaned_up_sparse_modes, sparse_context.rank_eigen_values);

                if ((position % sparse_context.num_procs) != (i % sparse_context.num_procs)) {
                    MPI_Send(cleaned_up_sparse_modes, sparse_context.rank_eigen_values, MPI_FLOAT, (i % sparse_context.num_procs), 0, MPI_COMM_WORLD);
                }
            }

            if ((i % sparse_context.num_procs) == sparse_context.my_rank) {
                if ((position % sparse_context.num_procs) != (i % sparse_context.num_procs)) {
                    MPI_Recv(cleaned_up_sparse_modes, sparse_context.rank_eigen_values, MPI_FLOAT, (position % sparse_context.num_procs), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                int col = i / sparse_context.num_procs;
                for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
                    sparse_context.sparse_modes[col * sparse_context.rank_eigen_values + j] = cleaned_up_sparse_modes[j];
                }
            }

            matrix_mul_1D_process_grid(sparse_context.sparse_modes, sparse_context.sparse_modes, &G, rows_G_local, cols_G_local, sparse_context.rank_eigen_values, sparse_context.num_modes, sparse_context.num_modes,\
                                       'T', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, true);
        }
    }

    deallocate(&condition);
    deallocate(&error_norm_vec);
    deallocate(&error_norm_vec_transpose);
    deallocate(&errors);
    deallocate(&G);
}
