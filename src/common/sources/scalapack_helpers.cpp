#include "../headers/utilities.h"
#include "../headers/log.h"
#include "../headers/scalapack_helpers.h"

/**
    Matrix multiplication in parallel

    @param void
    @return void
*/
void matrix_mul_1D_process_grid(float *A, float *B, float **C, int& rows_C_local, int& cols_C_local, int m, int k, int n, \
                                char transpose_A, char transpose_B, int nprow, int npcol, int mb_A, int nb_A, int mb_B, int nb_B, bool is_A_same_as_B) {
    /* Parameters for pdgemm*/
    float zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    /*  Local scalars */
    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT info, lld_A, lld_B, lld_C;
    MDESC desc_A, desc_B, desc_C;

    // Get information about how many processes are used for program execution and number of current process
    blacs_pinfo_( &iam, &nprocs );

    SCALAPACK_LOGD("(In matrix_mul_1D) iam", iam);
    SCALAPACK_LOGD("(In matrix_mul_1D) nprocs", nprocs);

    // Init working 1D process grid
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In matrix_mul_1D) myrow", myrow);
    SCALAPACK_LOGD("(In matrix_mul_1D) mycol", mycol);

    if (is_A_same_as_B) {
        // compute leading dimension of matrices
        lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
        lld_B = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
        SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

        // initializing array descriptors
        descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
        descinit_( desc_B, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    }

    if (transpose_A == 'N' && transpose_B == 'N') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &k, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &mb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &n, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        psgemm_(&transpose_A, &transpose_B, &m, &n, &k, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);

    } else if (transpose_A == 'T' && transpose_B == 'N') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &m, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &k, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &m, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &k, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &n, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &k, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &k, &n, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        psgemm_(&transpose_A, &transpose_B, &k, &n, &m, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    } else if (transpose_A == 'N' && transpose_B == 'T') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &k, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &k, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        psgemm_(&transpose_A, &transpose_B, &m, &k, &n, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    } else {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &k, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &k, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        psgemm_(&transpose_A, &transpose_B, &m, &k, &n, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    }

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}

/**
    Matrix transpose in parallel

    @param void
    @return void
*/
void matrix_transpose(float *A, int m, int n, float **C, int& rows_C_local, int& cols_C_local, int nprow, int npcol, int mb_A, int nb_A, int mb_C, int nb_C) {
    float zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    MDESC descA, descC;

    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT info, lld_A, lld_C;

    // Get information about how many processes are used for program execution and number of current process
    blacs_pinfo_( &iam, &nprocs );

    SCALAPACK_LOGD("(In matrix_transpose) iam", iam);
    SCALAPACK_LOGD("(In matrix_transpose) nprocs", nprocs);

    // Init working 1D process grid
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In matrix_transpose) myrow", myrow);
    SCALAPACK_LOGD("(In matrix_transpose) mycol", mycol);

    // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
    rows_C_local = numroc_( &n, &mb_C, &myrow, &i_zero, &nprow );
    cols_C_local = numroc_( &m, &nb_C, &mycol, &i_zero, &npcol );

    SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
    SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

    // allocating memory for output matrix C
    allocate(C, rows_C_local * cols_C_local);

    // compute leading dimension of matrices
    lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
    lld_C = MAX(1, numroc_( &n, &mb_C, &myrow, &i_zero, &nprow ));

    SCALAPACK_LOGD("(In matrix_transpose) lld_A", lld_A);
    SCALAPACK_LOGD("(In matrix_transpose) lld_C", lld_C);

    descinit_( descA, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    descinit_( descC, &n, &m, &mb_C, &nb_C, &i_zero, &i_zero, &ictxt, &lld_C, &info );

    pstran_(&n, &m, &one, A, &i_one, &i_one,descA, &zero, *C, &i_one, &i_one, descC);

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}

/**
    Singular value decomposition in parallel

    @param void
    @return void
*/
void singular_value_decomposition(float *A, char jobu, char jobvt, int m, int n, float **eigen_values, float **U, float **Vt, int& size_of_eigen_values, int& rows_U_local, int& cols_U_local,\
                                  int& rows_Vt_local, int& cols_Vt_local, int nprow, int npcol, int mb_A, int nb_A, int mb_U, int nb_U, int mb_Vt, int nb_Vt) {
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    MDESC descA, descU, descVt;

    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT info, lld_A, lld_U, lld_Vt;

    float *work = NULL;
    float workpt;
    MKL_INT lwork = -1;

    blacs_pinfo_( &iam, &nprocs );
    SCALAPACK_LOGD("(In singular_value_decomposition) iam", iam);
    SCALAPACK_LOGD("(In singular_value_decomposition) nprocs", nprocs);

    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In singular_value_decomposition) myrow", myrow);
    SCALAPACK_LOGD("(In singular_value_decomposition) mycol", mycol);

    rows_U_local = numroc_( &m, &mb_U, &myrow, &i_zero, &nprow );
    cols_U_local = numroc_( &n, &nb_U, &mycol, &i_zero, &npcol );

    SCALAPACK_LOGD("(In singular_value_decomposition) rows_U_local", rows_U_local);
    SCALAPACK_LOGD("(In singular_value_decomposition) cols_U_local", cols_U_local);

    rows_Vt_local = numroc_( &n, &mb_Vt, &myrow, &i_zero, &nprow );
    cols_Vt_local = numroc_( &n, &nb_Vt, &myrow, &i_zero, &npcol );

    SCALAPACK_LOGD("(In singular_value_decomposition) rows_Vt_local", rows_Vt_local);
    SCALAPACK_LOGD("(In singular_value_decomposition) cols_Vt_local", cols_Vt_local);

    // compute leading dimension of matrices
    lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
    lld_U = MAX(1, numroc_( &m, &mb_U, &myrow, &i_zero, &nprow));
    lld_Vt = MAX(1, numroc_( &n, &mb_Vt, &myrow, &i_zero, &nprow));

    SCALAPACK_LOGD("(In singular_value_decomposition) lld_A", lld_A);
    SCALAPACK_LOGD("(In singular_value_decomposition) lld_U", lld_U);
    SCALAPACK_LOGD("(In singular_value_decomposition) lld_Vt", lld_Vt);

    size_of_eigen_values = MIN(m, n);
    allocate(eigen_values, size_of_eigen_values);

    if (jobu == 'V') {
        allocate(U, rows_U_local * cols_U_local);
    }

    if (jobvt == 'V') {
        allocate(Vt, rows_Vt_local * cols_Vt_local);
    }

    descinit_( descA, &m, &n, &nb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    descinit_( descU, &m, &n, &nb_U, &nb_U, &i_zero, &i_zero, &ictxt, &lld_U, &info );
    descinit_( descVt, &n, &n, &nb_Vt, &nb_Vt, &i_zero, &i_zero, &ictxt, &lld_Vt, &info );

    psgesvd_(&jobu, &jobvt, &m, &n, A, &i_one, &i_one, descA, *eigen_values, *U, &i_one, &i_one, descU, *Vt, &i_one, &i_one, descVt, &workpt, &lwork, &info);

    SCALAPACK_LOGD("workpt", workpt);

    lwork = (int) workpt;
    allocate(&work, lwork);

    psgesvd_(&jobu, &jobvt, &m, &n, A, &i_one, &i_one, descA, *eigen_values, *U, &i_one, &i_one, descU, *Vt, &i_one, &i_one, descVt, work, &lwork, &info);

    SCALAPACK_LOGD("info", info);

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );

    deallocate(&work);
}

void matrix_vector_mul(float *A, int m, int n, float *X, float **Y, int& out_vector_length, char transpose_A, int mb_A, int nb_A, int nprow, int npcol, int ia, int ja, int ix, int jx, int iy, int jy, int incx, int incy) {
    // TODO: this implementation works for a distributed matrix but the input and output vector in only a single process
    // Implementation where vector is distribted across procs is not taken care
    // This is implementation also assumes 1D cyclic distribution of data with procs along the columns

    float zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    MDESC descA, descX, descY;

    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT lld_A, lld_X, lld_Y, length_X, length_Y, info;

    // Get information about how many processes are used for program execution and number of current process
    blacs_pinfo_( &iam, &nprocs );

    SCALAPACK_LOGD("(In matrix_vector_mul) iam", iam);
    SCALAPACK_LOGD("(In matrix_vector_mul) nprocs", nprocs);

    // Init working 1D process grid
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In matrix_vector_mul) myrow", myrow);
    SCALAPACK_LOGD("(In matrix_vector_mul) mycol", mycol);

    if (transpose_A == 'N') {
        length_X = n;
        length_Y = m;
    } else {
        length_X = m;
        length_Y = n;
    }

    SCALAPACK_LOGD("length of X", length_X);
    SCALAPACK_LOGD("length of Y", length_Y);

    // allocating memory for output vector y
    out_vector_length = length_Y;
    allocate(Y, length_Y);

    // compute leading dimension of matrices
    lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
    lld_X = MAX(1, numroc_( &length_X, &i_one, &myrow, &i_zero, &nprow ));
    lld_Y = MAX(1, numroc_( &length_Y, &i_one, &myrow, &i_zero, &nprow ));

    // initializing array descriptors
    descinit_( descA, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    descinit_( descX, &length_X, &npcol, &i_one, &i_one, &i_zero, &i_zero, &ictxt, &lld_X, &info );
    descinit_( descY, &length_Y, &npcol, &i_one, &i_one, &i_zero, &i_zero, &ictxt, &lld_Y, &info );

    SCALAPACK_LOGD("(In matrix_vector_mul) lld_A", lld_A);
    SCALAPACK_LOGD("(In matrix_vector_mul) lld_X", lld_X);
    SCALAPACK_LOGD("(In matrix_vector_mul) lld_Y", lld_Y);

    psgemv_(&transpose_A, &m, &n, &one, A, &ia, &ja, descA, X, &ix, &jx, descX, &incx, &zero, *Y, &iy, &jy, descY, &incy);

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}

void matrix_mul_1D_process_grid(double *A, double *B, double **C, int& rows_C_local, int& cols_C_local, int m, int k, int n, \
                                char transpose_A, char transpose_B, int nprow, int npcol, int mb_A, int nb_A, int mb_B, int nb_B, bool is_A_same_as_B) {
    /* Parameters for pdgemm*/
    double zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    /*  Local scalars */
    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT info, lld_A, lld_B, lld_C;
    MDESC desc_A, desc_B, desc_C;

    // Get information about how many processes are used for program execution and number of current process
    blacs_pinfo_( &iam, &nprocs );

    SCALAPACK_LOGD("(In matrix_mul_1D) iam", iam);
    SCALAPACK_LOGD("(In matrix_mul_1D) nprocs", nprocs);

    // Init working 1D process grid
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In matrix_mul_1D) myrow", myrow);
    SCALAPACK_LOGD("(In matrix_mul_1D) mycol", mycol);

    if (is_A_same_as_B) {
        // compute leading dimension of matrices
        lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
        lld_B = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
        SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

        // initializing array descriptors
        descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
        descinit_( desc_B, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    }

    if (transpose_A == 'N' && transpose_B == 'N') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &k, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &mb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &n, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        pdgemm_(&transpose_A, &transpose_B, &m, &n, &k, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);

    } else if (transpose_A == 'T' && transpose_B == 'N') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &m, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &k, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &m, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &k, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &n, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &k, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &k, &n, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        pdgemm_(&transpose_A, &transpose_B, &k, &n, &m, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    } else if (transpose_A == 'N' && transpose_B == 'T') {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &k, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &k, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        pdgemm_(&transpose_A, &transpose_B, &m, &k, &n, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    } else {
        if (!is_A_same_as_B) {
            // compute leading dimension of matrices
            lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
            lld_B = MAX(1, numroc_( &k, &mb_B, &myrow, &i_zero, &nprow));

            SCALAPACK_LOGD("(In matrix_mul_1D) lld_A", lld_A);
            SCALAPACK_LOGD("(In matrix_mul_1D) lld_B", lld_B);

            // initializing array descriptors
            descinit_( desc_A, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
            descinit_( desc_B, &k, &n, &mb_B, &nb_B, &i_zero, &i_zero, &ictxt, &lld_B, &info );
        }

        // Compute precise length of local pieces and allocate array on each process for parts of distributed vectors
        rows_C_local = numroc_( &m, &nb_A, &myrow, &i_zero, &nprow );
        cols_C_local = numroc_( &k, &nb_B, &mycol, &i_zero, &npcol );

        SCALAPACK_LOGD("(In matrix_mul_1D) rows_C_local", rows_C_local);
        SCALAPACK_LOGD("(In matrix_mul_1D) cols_C_local", cols_C_local);

        // allocating memory for output matrix C
        allocate(C, rows_C_local * cols_C_local);

        // compute leading dimension of matrix C
        lld_C = MAX(1, numroc_( &m, &nb_A, &myrow, &i_zero, &nprow ));

        SCALAPACK_LOGD("(In matrix_mul_1D) lld_C", lld_C);

        // initializing array descriptors for matrix C
        descinit_( desc_C, &m, &k, &nb_A, &nb_B, &i_zero, &i_zero, &ictxt, &lld_C, &info );

        // calling parallel matrix multiplication in mkl_scalapack library
        pdgemm_(&transpose_A, &transpose_B, &m, &k, &n, &one, A, &i_one, &i_one, desc_A,\
                B, &i_one, &i_one, desc_B, &zero,\
                *C, &i_one, &i_one, desc_C);
    }

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}

void matrix_vector_mul(double *A, int m, int n, double *X, double **Y, int& out_vector_length, char transpose_A, int mb_A, int nb_A, int nprow, int npcol, int ia, int ja, int ix, int jx, int iy, int jy, int incx, int incy) {
    // TODO: this implementation works for a distributed matrix but the input and output vector in only a single process
    // Implementation where vector is distribted across procs is not taken care
    // This is implementation also assumes 1D cyclic distribution of data with procs along the columns

    double zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;

    MDESC descA, descX, descY;

    MKL_INT iam, nprocs, ictxt, myrow, mycol;
    MKL_INT lld_A, lld_X, lld_Y, length_X, length_Y, info;

    // Get information about how many processes are used for program execution and number of current process
    blacs_pinfo_( &iam, &nprocs );

    SCALAPACK_LOGD("(In matrix_vector_mul) iam", iam);
    SCALAPACK_LOGD("(In matrix_vector_mul) nprocs", nprocs);

    // Init working 1D process grid
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    SCALAPACK_LOGD("(In matrix_vector_mul) myrow", myrow);
    SCALAPACK_LOGD("(In matrix_vector_mul) mycol", mycol);

    if (transpose_A == 'N') {
        length_X = n;
        length_Y = m;
    } else {
        length_X = m;
        length_Y = n;
    }

    SCALAPACK_LOGD("length of X", length_X);
    SCALAPACK_LOGD("length of Y", length_Y);

    // allocating memory for output vector y
    out_vector_length = length_Y;
    allocate(Y, length_Y);

    // compute leading dimension of matrices
    lld_A = MAX(1, numroc_( &m, &mb_A, &myrow, &i_zero, &nprow ));
    lld_X = MAX(1, numroc_( &length_X, &i_one, &myrow, &i_zero, &nprow ));
    lld_Y = MAX(1, numroc_( &length_Y, &i_one, &myrow, &i_zero, &nprow ));

    // initializing array descriptors
    descinit_( descA, &m, &n, &mb_A, &nb_A, &i_zero, &i_zero, &ictxt, &lld_A, &info );
    descinit_( descX, &length_X, &npcol, &i_one, &i_one, &i_zero, &i_zero, &ictxt, &lld_X, &info );
    descinit_( descY, &length_Y, &npcol, &i_one, &i_one, &i_zero, &i_zero, &ictxt, &lld_Y, &info );

    SCALAPACK_LOGD("(In matrix_vector_mul) lld_A", lld_A);
    SCALAPACK_LOGD("(In matrix_vector_mul) lld_X", lld_X);
    SCALAPACK_LOGD("(In matrix_vector_mul) lld_Y", lld_Y);

    pdgemv_(&transpose_A, &m, &n, &one, A, &ia, &ja, descA, X, &ix, &jx, descX, &incx, &zero, *Y, &iy, &jy, descY, &incy);

    // Destroy temporary process grid
    blacs_gridexit_( &ictxt );
}
