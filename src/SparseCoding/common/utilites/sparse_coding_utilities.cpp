#include "../../../common/headers/log.h"
#include "../../../common/headers/scalapack_helpers.h"
#include "../../sparse_coding.h"

void SparseCoding::eigen_power(float *A, int rows, int cols, float eig_epsilon, float *eig_vec_l, float *eig_vec_r, float* eig_val_A) {
    LOGR("=========== eigen_power serial ===========", sparse_context.my_rank, sparse_context.master);

    double conv_eig_val = 1, conv_eig_vec = 1, eig_val = 100, eig_val_old;
    int iteration = 1, incx = 1, incy = 1;
    double alpha = 1.0, beta = 0.0;
    double *A_ = NULL, *B = NULL, *eig_vec = NULL, *eig_vec_old = NULL, *y = NULL, *eig_vec_temp = NULL, *eig_vec_l_dup = NULL, *eig_vec_r_dup = NULL;

    allocate(&A_, rows * cols);
    allocate(&B, cols * cols);
    allocate(&eig_vec, cols);
    allocate(&eig_vec_old, cols);
    allocate(&y, cols);
    allocate(&eig_vec_temp, cols);
    allocate(&eig_vec_l_dup, rows);
    allocate(&eig_vec_r_dup, cols);

    duplicate(A, A_, rows * cols);

    LOG("=============== A_ ============================");
    display(A_, rows * cols);

    dgemm("T", "N", &cols, &cols, &rows, &alpha, A_, &rows, A_, &rows, &beta, B, &cols);

    LOG("=============== B = A' * A ============================");
    display(B, cols * cols);

    for (int i = 0; i < cols; i++) {
        eig_vec[i] = rand()/double(RAND_MAX);
    }

    LOG("=============== random eig_vec ============================");
    display(eig_vec, cols);

    while (((conv_eig_val > eig_epsilon) || (conv_eig_vec > eig_epsilon)) && iteration < 100005) {
        eig_val_old = eig_val;
        duplicate(eig_vec, eig_vec_old, cols);

        dgemv("N", &cols, &cols, &alpha, B, &cols, eig_vec, &incx, &beta, y, &incy);

        LOG("=============== y ============================");
        display(y, cols);

        LOG("=============== eig_vec ============================");
        display(eig_vec, cols);

        eig_val = norm(eig_vec, cols);

        LOGD("eigen_val", eig_val);

        for (int i = 0; i < cols; i++) {
            eig_vec[i] = y[i] / eig_val;
        }

        LOG("=============== eig_vec after normalization ============================");
        display(eig_vec, cols);


        conv_eig_val = fabs((eig_val/eig_val_old) - 1.0);

        LOGD("conv_eig_val", conv_eig_val);

        for(int i = 0; i < cols; i++) {
            eig_vec_temp[i] = eig_vec[i] - eig_vec_old[i];
        }

        conv_eig_vec = norm(eig_vec_temp, cols) / norm(eig_vec_old, cols);

        LOGD("conv_eig_vec", conv_eig_vec);

        iteration += 1;
    }

    if (iteration > 100000) {
        LOG("EIG POWER FAILED TO CONVERGE");
    }

    for (int i = 0; i < cols; i++) {
        eig_vec[i] = eig_vec[i] * (1 / eig_val);
    }

    dgemv("N", &rows, &cols, &alpha, A_, &rows, eig_vec, &incx, &beta, eig_vec_l_dup, &incy);
    normalize(eig_vec_l_dup, rows);

    *eig_val_A = (float) sqrt(eig_val);

    duplicate(eig_vec, eig_vec_r_dup, cols);
    normalize(eig_vec_r_dup, cols);

    duplicate(eig_vec_l_dup, eig_vec_l, rows);
    duplicate(eig_vec_r_dup, eig_vec_r, cols);

    deallocate(&B);
    deallocate(&eig_vec);
    deallocate(&eig_vec_old);
    deallocate(&y);
    deallocate(&eig_vec_temp);
    deallocate(&eig_vec_l_dup);
    deallocate(&eig_vec_r_dup);
    deallocate(&A_);
}

void SparseCoding::eigen_power_parallel(float *A, int rows_A_local, int cols_A_local, int rows_A_global, int cols_A_global, float eig_epsilon, float *eig_vec_l, float *eig_vec_r, float *eig_val_A, int my_rank) {
    LOGR("=========== eigen_power_parallel ===========", sparse_context.my_rank, sparse_context.master);

    int rows_B_local, cols_B_local, length_of_y, iteration = 1, convergence_criteria = 2;
    double conv_eig_val = 1, conv_eig_vec = 1, eig_val = 100, eig_val_old;
    double *A_ = NULL, *B = NULL, *eig_vec = NULL, *eig_vec_old = NULL, *y = NULL, *eig_vec_temp = NULL, *eig_vec_l_dup = NULL, *convergence_values = NULL;

    allocate(&convergence_values, convergence_criteria);
    convergence_values[0] = conv_eig_val;
    convergence_values[1] = conv_eig_vec;

    allocate(&A_, rows_A_local * cols_A_local);
    duplicate(A, A_, rows_A_local * cols_A_local);

    if (sparse_context.my_rank == my_rank) {
        allocate(&eig_vec, rows_A_global);
        allocate(&eig_vec_old, rows_A_global);
        allocate(&eig_vec_temp, rows_A_global);
    }

    //matrix_mul_1D_process_grid(A, A, &B, rows_B_local, cols_B_local, rows, cols, cols, 'T', 'N', 1, sparse_context.num_procs, 1, 1, 1, 1, true);
    matrix_mul_1D_process_grid(A_, A_, &B, rows_B_local, cols_B_local, rows_A_global, rows_A_global, cols_A_global, 'N', 'T', 1, sparse_context.num_procs, 1, 1, 1, 1, true);

    SCALAPACK_LOG("=============== B = A' * A ============================");
    display(B, rows_B_local * cols_B_local);

    if (sparse_context.my_rank == my_rank) {
        for (int i = 0; i < rows_A_global; i++) {
            eig_vec[i] = rand()/double(RAND_MAX);
        }

        SCALAPACK_LOG("=============== eig_vec ============================");
        display(eig_vec, rows_A_global);
    }

    while (((convergence_values[0] > eig_epsilon) || (convergence_values[1] > eig_epsilon)) && iteration < 100005) {
        matrix_vector_mul(B, rows_A_global, rows_A_global, eig_vec, &y, length_of_y, 'N', 1, 1, 1, sparse_context.num_procs, 1, 1, 1, (my_rank+1), 1, (my_rank+1), 1, 1);

        if (sparse_context.my_rank == my_rank) {
            eig_val_old = eig_val;

            duplicate(eig_vec, eig_vec_old, rows_A_global);

            SCALAPACK_LOG("=============== y ============================");
            display(y, rows_A_global);

            eig_val = norm(eig_vec, rows_A_global);

            for (int i = 0; i < rows_A_global; i++) {
                eig_vec[i] = y[i] / eig_val;
            }

            SCALAPACK_LOG("=============== eig_vec after normalization ============================");
            display(eig_vec, rows_A_global);

            conv_eig_val = fabs((eig_val/eig_val_old) - 1.0);

            for(int i = 0; i < rows_A_global; i++) {
                eig_vec_temp[i] = eig_vec[i] - eig_vec_old[i];
            }

            conv_eig_vec = norm(eig_vec_temp, rows_A_global) / norm(eig_vec_old, rows_A_global);

            convergence_values[0] = conv_eig_val;
            convergence_values[1] = conv_eig_vec;

            SCALAPACK_LOGD("rank", sparse_context.my_rank);
            SCALAPACK_LOGD("conv_eig_val", convergence_values[0]);
            SCALAPACK_LOGD("conv_eig_vec", convergence_values[1]);
        }

        MPI_Bcast(convergence_values, convergence_criteria, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
        iteration += 1;
    }

    if (iteration > 100000) {
        LOG("EIG POWER FAILED TO CONVERGE");
    }

    if (sparse_context.my_rank == my_rank) {
        for (int i = 0; i < rows_A_global; i++) {
            eig_vec[i] = eig_vec[i] * (1 / eig_val);
        }
    }

    int eig_vec_l_length;

    matrix_vector_mul(A_, rows_A_global, cols_A_global, eig_vec, &eig_vec_l_dup, eig_vec_l_length, 'T', 1, 1, 1, sparse_context.num_procs, 1, 1, 1, (my_rank+1), 1, (my_rank+1), 1, 1);

    if (sparse_context.my_rank == my_rank) {
        normalize(eig_vec_l_dup, eig_vec_l_length);

        SCALAPACK_LOG("=============== eig_vec_l_dup ============================");
        display(eig_vec_l_dup, eig_vec_l_length);

        duplicate(eig_vec_l_dup, sparse_context.updated_modes, eig_vec_l_length);

        *eig_val_A = (float) sqrt(eig_val);
        SCALAPACK_LOGD("eig_val_A", *eig_val_A);

        normalize(eig_vec, rows_A_global);
        duplicate(eig_vec, eig_vec_r, rows_A_global);

        SCALAPACK_LOG("=============== eig_vec_r ============================");
        display(eig_vec_r, rows_A_global);
    }

    deallocate(&convergence_values);
    deallocate(&eig_vec);
    deallocate(&eig_vec_old);
    deallocate(&eig_vec_temp);
    deallocate(&y);
    deallocate(&eig_vec_l_dup);
    deallocate(&B);
    deallocate(&A_);
}

// This function will not work properly for sub rank matrices where we encouter 'nan' and 'inf' values for PinvS since S will be super small and sometimes 0
void SparseCoding::pseudo_inverse(float *A, int& m, int& n) {
    LOGR("=========== pseudo_inverse serial ===========", sparse_context.my_rank, sparse_context.master);

    // SVD of A to get U, Vt and S
    int lda = m, ldu = m, ldvt = n, info, lwork;
    float wkopt;
    float* work = NULL;
    float *S = NULL, *U = NULL, *Vt = NULL, *PinvS = NULL, *C = NULL;
    float alpha = 1.0, beta = 0.0;

    allocate(&S, n);
    allocate(&U, ldu * m);
    allocate(&Vt, ldvt * n);
    allocate(&PinvS, n * m);
    allocate(&C, n * m);

    lwork = -1;
    sgesvd( "All", "All", &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, &wkopt, &lwork, &info );

    lwork = (int) wkopt;
    allocate(&work, lwork);
    sgesvd("All", "All", &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, work, &lwork, &info);

    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        MPI_Finalize();
        exit(1);
    }

    SCALAPACK_LOG("=============== U ============================");
    display(U, ldu * m);
    SCALAPACK_LOG("=============== S ============================");
    display(S, n);
    SCALAPACK_LOG("=============== Vt ============================");
    display(Vt, ldvt * n);

    // inverse of S
    for (int i = 0; i < n; i++) {
        S[i] = 1/S[i];
    }

    // Form Pinv(A) using V * inv(S) * Ut
    for (int i = 0; i < n * m; i++) {
        PinvS[i] = 0;
    }

    for (int i = 0, j = 0; i < n * m; i+= (m+1), j++) {
        PinvS[i] = S[j];
    }

    SCALAPACK_LOG("=============== PinvS ============================");
    display(PinvS, n*m);

    // calculating PinvS * Ut
    sgemm("T", "T", &n, &m, &m, &alpha, PinvS, &m, U, &m, &beta, C, &n);

    SCALAPACK_LOG("=============== C ============================");
    display(C, n*m);

    sgemm("T", "N", &n, &m, &n, &alpha, Vt, &n, C, &n, &beta, A, &n);

    deallocate(&work);
    deallocate(&S);
    deallocate(&U);
    deallocate(&Vt);
    deallocate(&PinvS);
    deallocate(&C);
}

// This function will not work properly for sub rank matrices where we encouter 'nan' and 'inf' values for PinvS since S will be super small and sometimes 0
void SparseCoding::pseudo_inverse(float *A, float **pinvA, int& m, int& n, int& rows_pinvA, int& cols_pinvA) {
    LOGR("=========== pseudo_inverse parallel ===========", sparse_context.my_rank, sparse_context.master);

    // SVD of A to get U, Vt and S
    float zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    MDESC descA, descU, descVt, descpinvA;

    MKL_INT iam, nprocs, ictxt, myrow, mycol, nprow, npcol;
    MKL_INT mb, nb, mp, nq;
    MKL_INT info, lld, lwork = -1;

    char jobu = 'V', jobvt = 'V';
    char transpose = 'T', no_transpose = 'N';
    float *work, *S = NULL, *U = NULL, *Vt = NULL, *Vt_into_S = NULL;
    float workpt;
    int remainder, num_cols_pinvA_per_rank;

    blacs_pinfo_( &iam, &nprocs );
    SCALAPACK_LOGD("(In psuedo_inverse) iam", iam);
    SCALAPACK_LOGD("(In pseudo_inverse) nprocs", nprocs);

    // TODO: currently using 1D block distribution of data
    // Need to change it to 2D block cyclic to test for efficiency
    nprow = 1;
    npcol = nprocs;

    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In psuedo_inverse) myrow", myrow);
    SCALAPACK_LOGD("(In psuedo_inverse) mycol", mycol);

    // TODO: need to fix these parameters by proc.txt file or understand better and fix in an intelligent manner
    mb = m;
    nb = 1; //sparse_context.num_snapshots / nprallocate(&pinvA, n * nq);ocs;

    mp = numroc_( &m, &mb, &myrow, &i_zero, &nprow );
    nq = numroc_( &n, &nb, &mycol, &i_zero, &npcol );

    lld = MAX( mp, 1 );

    SCALAPACK_LOGD("(In psuedo_inverse) mp", mp);
    SCALAPACK_LOGD("(In psuedo_inverse) nq", nq);

    allocate(&S, n);
    allocate(&U, m * nq);
    allocate(&Vt, n * nq);
    allocate(&Vt_into_S, n * nq);

    descinit_( descA, &m, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
    descinit_( descU, &m, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &m, &info );
    descinit_( descVt, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &n, &info );

    psgesvd(&jobu, &jobvt, &m, &n, A, &i_one, &i_one, descA, S, U, &i_one, &i_one, descU, Vt, &i_one, &i_one, descVt, &workpt, &lwork, &info);

    SCALAPACK_LOGD("workpt", workpt);

    lwork = (int)workpt;
    work = (float*) calloc(lwork, sizeof( float ));

    psgesvd(&jobu, &jobvt, &m, &n, A, &i_one, &i_one, descA, S, U, &i_one, &i_one, descU, Vt, &i_one, &i_one, descVt, work, &lwork, &info);

    SCALAPACK_LOG("=============== S ============================");
    display(S, n);
    SCALAPACK_LOG("=============== U ============================");
    display(U, m * nq);
    SCALAPACK_LOG("=============== Vt ============================");
    display(Vt, n * nq);

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < n; j++) {
            Vt_into_S[i * n + j] = Vt[i * n + j] * (1 / S[j]);
        }
    }

    SCALAPACK_LOG("=============== Vt_into_S ============================");
    display(Vt_into_S, n * nq);

    // PinvA is essentially the transpose to A in terms of matrix dimensions
    // Since A is distributed (m*n) with m as rows and say there was one row for each process previously,
    // there will be more than one row in one of the processes with pinvA
    // for ex: 3 procs and dim of A: m = 4, n = 3. so each proc has a vector of size 4 now
    // dim of pinvA: m = 3, n= 4. Therefore data in 3 procs has to be distributed as
    // proc1: m = 3, n = 2; m = 3, n = 1, m = 3, n = 1. Below code to find remainder and num of cols does the same

    remainder = m % sparse_context.num_procs;
    num_cols_pinvA_per_rank = m / sparse_context.num_procs;

    if (sparse_context.my_rank < remainder) {
        num_cols_pinvA_per_rank++;
    }

    rows_pinvA = n;
    cols_pinvA = num_cols_pinvA_per_rank;

    allocate(pinvA, n * num_cols_pinvA_per_rank);
    descinit_( descpinvA, &n, &m, &nb, &nb, &i_zero, &i_zero, &ictxt, &n, &info );

    psgemm_(&transpose, &transpose, &n , &m, &n, &one, Vt_into_S, &i_one, &i_one, descVt, U, &i_one, &i_one, descU, &zero, *pinvA, &i_one, &i_one, descpinvA);

    deallocate(&work);
    deallocate(&S);
    deallocate(&U);
    deallocate(&Vt);
    deallocate(&Vt_into_S);

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}
