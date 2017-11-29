#include "../../../common/headers/log.h"
#include "../../../common/headers/scalapack_helpers.h"
#include "../../sparse_coding.h"


void SparseCoding::generate_sparse_modes_serial() {
    LOGR("================= generate_sparse_modes =======================", sparse_context.my_rank, sparse_context.master);

    int iteration = 0;
    float epsilon_val = 1e10, error = 0, prev_error;
    float alpha = 1.0, beta = 0.0;
    float *error_matrix = NULL;
    double start_time_convergence_loop;
    allocate(&sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.num_snapshots);
    allocate(&error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

    while (epsilon_val > sparse_context.epsilon && iteration < sparse_context.max_iterations) {
        start_time_convergence_loop = MPI_Wtime();

        batch_OMP_serial();
        measure_time_for_function(start_time_convergence_loop, "Batch OMP parallel done", sparse_context.my_rank, sparse_context.master);
        compute_memory_readings(sparse_context.num_procs, sparse_context.my_rank, sparse_context.master);

        KSVD_serial();
        measure_time_for_function(start_time_convergence_loop, "KSVD parallel done", sparse_context.my_rank, sparse_context.master);
        compute_memory_readings(sparse_context.num_procs, sparse_context.my_rank, sparse_context.master);

        I_clear_dictionary_serial();
        measure_time_for_function(start_time_convergence_loop, "I clear dictionary done", sparse_context.my_rank, sparse_context.master);
        compute_memory_readings(sparse_context.num_procs, sparse_context.my_rank, sparse_context.master);

        error = 0;
        sgemm("N", "N", &sparse_context.rank_eigen_values, &sparse_context.num_snapshots, &sparse_context.num_modes, &alpha, sparse_context.sparse_modes, &sparse_context.rank_eigen_values,\
              sparse_context.coefficient_matrix, &sparse_context.num_modes, &beta, error_matrix, &sparse_context.rank_eigen_values);

        LOG("=============== sparse_modes * coefficient_matrix ============================");
        display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

        for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
            error_matrix[i] = sparse_context.pod_coefficients[i] - error_matrix[i];
        }

        LOG("=============== Error matrix ============================");
        display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

        for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
            error_matrix[i] = error_matrix[i] * error_matrix[i];
        }

        for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
            error += error_matrix[i] / (sparse_context.rank_eigen_values * sparse_context.num_snapshots);
        }

        error = sqrt(error);

        if (iteration > 5) {
            epsilon_val = fabs(prev_error - error);
        }

        LOGDR_("prev error", prev_error, sparse_context.my_rank, sparse_context.master);
        LOGDR_("error", error, sparse_context.my_rank, sparse_context.master);
        LOGDR("epsilon val", epsilon_val, sparse_context.my_rank, sparse_context.master);

        prev_error = error;
        iteration += 1;
       
        LOGDR("iteration", iteration, sparse_context.my_rank, sparse_context.master);
        measure_time_for_function(start_time, "Time after current iteration", sparse_context.my_rank, sparse_context.master);
        LOGR("******************************************************************************************************", sparse_context.my_rank, sparse_context.master);
    }

    LOGDR("Total iterations", iteration, sparse_context.my_rank, sparse_context.master);

    deallocate(&error_matrix);

    // deallocating right after use to reduce peak mem usage
    if(!(sparse_context.is_write_sparse_transformation_matrix || sparse_context.is_write_sparse_modes_in_original_domain || sparse_context.is_write_corrected_sparse_coefficients || sparse_context.is_write_sparse_reconstruction_error))
        deallocate((&sparse_context.sparse_modes));

    // deallocating right after use to reduce peak mem usage
    if(!sparse_context.is_write_sparse_coefficients)
        deallocate(&sparse_context.coefficient_matrix);

}

void SparseCoding::batch_OMP_serial() {
    LOG("=============== Batch OMP ============================");

    float alpha = 1.0, beta = 0.0, max = 0.0;
    int *index = NULL;
    float *G = NULL, *x = NULL, *y = NULL, *y_dup = NULL, *y_ = NULL, *PinvG = NULL;
    float *PinvG_prod_y_ = NULL, *G_ = NULL, *G_prod_PinvG_prod_y_ = NULL, *D_ = NULL, *A_ = NULL;
    int incx = 1, incy = 1, m , pos;

    allocate(&G, sparse_context.num_modes * sparse_context.num_modes);
    allocate(&x, sparse_context.rank_eigen_values);
    allocate(&y, sparse_context.num_modes);
    allocate(&y_dup, sparse_context.num_modes);
    allocate(&index, sparse_context.num_bases_active);

    initialize(sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.num_snapshots);

    sgemm("T", "N", &sparse_context.num_modes, &sparse_context.num_modes, &sparse_context.rank_eigen_values, &alpha, sparse_context.sparse_modes,\
          &sparse_context.rank_eigen_values, sparse_context.sparse_modes, &sparse_context.rank_eigen_values, &beta, G, &sparse_context.num_modes);
    display(G, sparse_context.rank_eigen_values * sparse_context.num_modes);

    for (int i = 0; i < sparse_context.num_snapshots; i++) {
        LOGD_("============================= Iterartion", i+1);
        LOG("=====================================");

        for (int k = 0; k < sparse_context.rank_eigen_values; k++) {
            x[k] = sparse_context.pod_coefficients[i * sparse_context.rank_eigen_values + k];
        }

        LOG("=============== x ============================");
        display(x, sparse_context.rank_eigen_values);

        sgemv("T", &sparse_context.rank_eigen_values, &sparse_context.num_modes, &alpha, sparse_context.sparse_modes, &sparse_context.rank_eigen_values, x, &incx, &beta, y, &incy);

        duplicate(y, y_dup, sparse_context.num_modes);
        LOG("=============== y ============================");
        
        for (int j = 0; j < sparse_context.num_bases_active; j++) {
            get_abs_max_and_pos(y, sparse_context.num_modes, &max, &pos);
            LOGD("max val", max);
            LOGD("pos", pos);

            index[j] = pos;

            m = j+1;
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
            LOG("=============== PinvG ============================");
            display(PinvG, m * m);

            LOG("=============== PinvG * y_ ============================");
            sgemv("N", &m, &m, &alpha, PinvG, &m, y_, &incx, &beta, PinvG_prod_y_, &incy);
            display(PinvG_prod_y_, m);

            // truncated G[0:m)[index[0:m]]
            for (int a = 0, c = 0; a < m; a++) {
                for (int b = 0; b < sparse_context.num_modes; b++, c++) {
                    G_[c] = G[index[a] * sparse_context.num_modes + b];
                }
            }

            LOG("=============== G_ ============================");
            display(G_, sparse_context.num_modes * m);

            LOG("=============== G_ * PinvG * y_ ============================");
            sgemv("N", &sparse_context.num_modes, &m, &alpha, G_, &sparse_context.num_modes, PinvG_prod_y_, &incx, &beta, G_prod_PinvG_prod_y_, &incy);
            display(G_prod_PinvG_prod_y_, sparse_context.num_modes);

            LOG("=============== updating y ============================");
            for (int a = 0; a < sparse_context.num_modes; a++) {
                y[a] = y_dup[a] - G_prod_PinvG_prod_y_[a];
            }
            display(y, sparse_context.num_modes);


            for (int a = 0; a < m; a++) {
                    y[index[a]] = 0;
            }

            deallocate(&PinvG);
            deallocate(&y_);
            deallocate(&PinvG_prod_y_);
            deallocate(&G_);
            deallocate(&G_prod_PinvG_prod_y_);
        }

        allocate(&D_, sparse_context.rank_eigen_values * sparse_context.num_bases_active);
        // truncated sparse_modes[0:num_snapshots)[index[0:num_bases_active]]
        for (int a = 0, c = 0; a < sparse_context.num_bases_active; a++) {
            for (int b = 0; b < sparse_context.rank_eigen_values; b++, c++) {
                D_[c] = sparse_context.sparse_modes[index[a] * sparse_context.rank_eigen_values + b];
            }
        }

        LOG("=============== D_ ============================");
        display(D_, sparse_context.rank_eigen_values * sparse_context.num_bases_active);

        pseudo_inverse(D_, sparse_context.rank_eigen_values, sparse_context.num_bases_active);
        LOG("=============== PinvD_ ============================");
        display(D_, sparse_context.num_bases_active * sparse_context.rank_eigen_values);

        LOG("=============== A_ ============================");
        allocate(&A_, sparse_context.num_bases_active);
        sgemv("N", &sparse_context.num_bases_active, &sparse_context.rank_eigen_values, &alpha, D_, &sparse_context.num_bases_active, x, &incx, &beta, A_, &incy);
        display(A_, sparse_context.num_bases_active);

        for (int a = 0; a < sparse_context.num_bases_active; a++) {
            sparse_context.coefficient_matrix[i * sparse_context.num_modes + index[a]] = A_[a];
        }

        deallocate(&D_);
        deallocate(&A_);
    }

    LOG("=============== coefficient matrix ============================");
    display(sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.num_snapshots);

    deallocate(&G);
    deallocate(&x);
    deallocate(&y);
    deallocate(&y_dup);
    deallocate(&index);
}

void SparseCoding::KSVD_serial() {
    LOG("=============== KSVD_serial ============================");

    allocate(&sparse_context.updated_modes, sparse_context.rank_eigen_values);

    for (int i = 0; i < sparse_context.num_modes; i++) {
        update_modes_KSVD_serial(i);

        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            sparse_context.sparse_modes[i * sparse_context.rank_eigen_values + j] = sparse_context.updated_modes[j];
        }
    }

    deallocate(&sparse_context.updated_modes);
}

void SparseCoding::update_modes_KSVD_serial(int &row) {
    LOG("=============== update_modes_KSVD_serial ============================");

    float singular_value = 0, *beta_vector = NULL, *coefficient_matrix_relevant_indices = NULL, *errors = NULL;
    float alpha = 1.0, beta = 0.0, epsilon = 1e-10;
    int size;

    vector<int> relevant_data_indices = get_indices_nonzero_each_row(row, sparse_context.coefficient_matrix, sparse_context.num_modes, sparse_context.num_snapshots);
    size = relevant_data_indices.size();
    LOGD("relevant indices size", relevant_data_indices.size());

    if (relevant_data_indices.size() < 1) {
        int pos=0;
        replace_non_active_mode_serial(row, pos);
        return;
    }

    allocate(&coefficient_matrix_relevant_indices, sparse_context.num_modes * relevant_data_indices.size());
    allocate(&errors, sparse_context.rank_eigen_values * relevant_data_indices.size());
    allocate(&beta_vector, relevant_data_indices.size());


    for (int i = 0; i < relevant_data_indices.size(); i++) {
        for (int j = 0; j < sparse_context.num_modes; j++) {
            coefficient_matrix_relevant_indices[i * sparse_context.num_modes + j] = sparse_context.coefficient_matrix[relevant_data_indices[i] * sparse_context.num_modes + j];
        }
    }

    LOG("=============== coefficient_matrix_relevant_indices ===========================");
    display(coefficient_matrix_relevant_indices, sparse_context.num_modes * relevant_data_indices.size());
    
    sgemm("N", "N", &sparse_context.rank_eigen_values, &size, &sparse_context.num_modes, &alpha, sparse_context.sparse_modes, \
          &sparse_context.rank_eigen_values, coefficient_matrix_relevant_indices, &sparse_context.num_modes, &beta, errors, &sparse_context.rank_eigen_values);

    LOG("=============== sparse_modes ===========================");
    display(sparse_context.sparse_modes, sparse_context.rank_eigen_values * sparse_context.num_modes);

    LOG("=============== sparse_modes * coefficient_matrix_relevant_indices ===========================");
    display(errors, sparse_context.rank_eigen_values * relevant_data_indices.size());

    for (int i = 0; i < relevant_data_indices.size(); i++) {
        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            errors[i * sparse_context.rank_eigen_values + j] = sparse_context.pod_coefficients[relevant_data_indices[i] * sparse_context.rank_eigen_values + j] -
                    errors[i * sparse_context.rank_eigen_values + j];
        }
    }

    round_off_below_machine_precision_to_zero(errors, sparse_context.rank_eigen_values * relevant_data_indices.size());

    LOG("=============== errors ===========================");
    display(errors, sparse_context.rank_eigen_values * relevant_data_indices.size());

    eigen_power(errors, sparse_context.rank_eigen_values, relevant_data_indices.size(), epsilon, sparse_context.updated_modes, beta_vector, &singular_value);

    LOG("=============== singular_value ===========================");
    LOGD("singular value", singular_value);

    LOG("=============== updated modes ===========================");
    display(sparse_context.updated_modes, sparse_context.rank_eigen_values);

    LOG("=============== beta vector ===========================");
    display(beta_vector, relevant_data_indices.size());

    for (int i = 0; i < relevant_data_indices.size(); i++) {
        sparse_context.coefficient_matrix[sparse_context.num_modes * relevant_data_indices[i] + row] = singular_value * beta_vector[i];
    }

    LOG("=============== coefficient matrix ============================");
    display(sparse_context.coefficient_matrix, sparse_context.num_modes * sparse_context.num_snapshots);

    deallocate(&coefficient_matrix_relevant_indices);
    deallocate(&errors);
    deallocate(&beta_vector);

}

void SparseCoding::replace_non_active_mode_serial(int row, int &pos) {
    LOG("================== replace_non_active_mode_serial =======================");

    float alpha = 1.0, beta = 0.0;
    float *error_norm_vec = NULL, *error_matrix = NULL, max;
    int sign_val;

    allocate(&error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);
    allocate(&error_norm_vec, sparse_context.num_snapshots);

    sgemm("N", "N", &sparse_context.rank_eigen_values, &sparse_context.num_snapshots, &sparse_context.num_modes, &alpha, sparse_context.sparse_modes, &sparse_context.rank_eigen_values,\
          sparse_context.coefficient_matrix, &sparse_context.num_modes, &beta, error_matrix, &sparse_context.rank_eigen_values);

    LOG("=============== sparse_modes * coefficient_matrix ============================");
    display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

    for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
        error_matrix[i] = sparse_context.pod_coefficients[i] - error_matrix[i];
    }


    for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
        error_matrix[i] = error_matrix[i] * error_matrix[i];
    }

    LOG("=============== Error matrix ============================");
    display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

    for (int i = 0; i < sparse_context.num_snapshots; i++) {
        error_norm_vec[i] = 0;
        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            error_norm_vec[i] += error_matrix[i * sparse_context.rank_eigen_values + j];
        }
    }

    get_max_and_pos(error_norm_vec, sparse_context.num_snapshots, &max, &pos);
    LOGD("max val", max);
    LOGD("pos", pos);

    for (int i = 0; i < sparse_context.rank_eigen_values; i++) {
        sparse_context.updated_modes[i] = sparse_context.pod_coefficients[pos * sparse_context.rank_eigen_values + i];
    }

    sparse_context.coefficient_matrix[pos * sparse_context.num_modes + row] = norm(sparse_context.updated_modes, sparse_context.rank_eigen_values);


    normalize(sparse_context.updated_modes, sparse_context.rank_eigen_values);

    // TODO: free at the end rather than here to avoid multiple allocation
    // However, overall memory will be high by doing that and hence need to evaluate
    deallocate(&error_matrix);
    deallocate(&error_norm_vec);

    return;
}

void SparseCoding::I_clear_dictionary_serial() {
    LOG("=============== I_clear_dictionary ============================");

    float T2 = 0.99, T1 = 3.0, alpha = 1.0, beta = 0.0;;
    float *error_norm_vec = NULL, *G = NULL, *replacement_vector = NULL, max_val;
    int position, num_of_vals_greater_than_threshold;
    float *error_matrix = NULL;

    allocate(&error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);
    allocate(&error_norm_vec, sparse_context.num_snapshots);
    allocate(&G, sparse_context.num_modes * sparse_context.num_modes);
    allocate(&replacement_vector, sparse_context.num_snapshots);

    initialize(error_norm_vec, sparse_context.num_snapshots);

    sgemm("N", "N", &sparse_context.rank_eigen_values, &sparse_context.num_snapshots, &sparse_context.num_modes, &alpha, sparse_context.sparse_modes, &sparse_context.rank_eigen_values,\
          sparse_context.coefficient_matrix, &sparse_context.num_modes, &beta, error_matrix, &sparse_context.rank_eigen_values);

    LOG("=============== sparse_modes * coefficient_matrix ============================");
    display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

    for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
        error_matrix[i] = sparse_context.pod_coefficients[i] - error_matrix[i];
    }

    LOG("=============== Error matrix ============================");
    display(error_matrix, sparse_context.rank_eigen_values * sparse_context.num_snapshots);

    for (int i = 0; i < sparse_context.rank_eigen_values * sparse_context.num_snapshots; i++) {
        error_matrix[i] = error_matrix[i] * error_matrix[i];
    }

    for (int i = 0; i < sparse_context.num_snapshots; i++) {
        for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
            error_norm_vec[i] += error_matrix[i * sparse_context.rank_eigen_values + j];
        }
    }
    
    sgemm("T", "N", &sparse_context.num_modes, &sparse_context.num_modes, &sparse_context.rank_eigen_values, &alpha, sparse_context.sparse_modes, \
          &sparse_context.rank_eigen_values, sparse_context.sparse_modes, &sparse_context.rank_eigen_values, &beta, G, &sparse_context.num_modes);

    // assigning diagonal elements to zero as the modes are correlated to itself
    for (int i = 0; i < sparse_context.num_modes * sparse_context.num_modes; i += (sparse_context.num_modes+1)) {
        G[i] = 0;
    }

    LOG("=============== G = sparse_modes' * sparse_modes ============================");
    display(G, sparse_context.num_modes * sparse_context.num_modes);

    for (int i = 0; i < sparse_context.num_modes; i++) {
        get_max_and_pos(&G[i * sparse_context.num_modes], sparse_context.num_modes, &max_val, &position);
        LOGD("max value", max_val);

        num_of_vals_greater_than_threshold = get_num_elements_greater_than_threshold_along_row(sparse_context.coefficient_matrix, \
                                             sparse_context.num_modes, sparse_context.num_snapshots, i, 1e-10);

        if (max_val > T2 || num_of_vals_greater_than_threshold <= T1) {

            get_max_and_pos(error_norm_vec, sparse_context.num_snapshots, &max_val, &position);
            LOGD("max value", max_val);
            LOGD("position", position);

            error_norm_vec[position] = 0;

            duplicate(&sparse_context.pod_coefficients[position * sparse_context.num_snapshots], replacement_vector, sparse_context.num_snapshots);
            normalize(replacement_vector, sparse_context.num_snapshots);

            for (int j = 0; j < sparse_context.rank_eigen_values; j++) {
                sparse_context.sparse_modes[i * sparse_context.rank_eigen_values + j] = replacement_vector[j];
            }

            sgemm("T", "N", &sparse_context.num_modes, &sparse_context.num_modes, &sparse_context.rank_eigen_values, &alpha, sparse_context.sparse_modes, \
                  &sparse_context.rank_eigen_values, sparse_context.sparse_modes, &sparse_context.rank_eigen_values, &beta, G, &sparse_context.num_modes);

            // assigning diagonal elements to zero as the modes are correlated to itself
            for (int j = 0; j < sparse_context.num_modes * sparse_context.num_modes; j += (sparse_context.num_modes+1)) {
                G[j] = 0;
            }
        }
    }

    deallocate(&error_matrix);
    deallocate(&error_norm_vec);
    deallocate(&G);
    deallocate(&replacement_vector);
}
