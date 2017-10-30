#include <iostream>
#include "test_scalapack.h"

using namespace std;

void test_dgemm() {
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    alpha = 1.0, beta = 0.0;
    m = 5, k = 3, n = 4;

    allocate(&A, m*k);
    allocate(&B, k*n);
    allocate(&C, m*n);

    for (int i = 0; i < m*k; i++) {
        A[i] = i+1;
    }

    for (int i = 0; i < k*n; i++) {
        B[i] = i+1;
    }

    for (int i = 0; i < m*n; i++) {
        C[i] = 0;
    }

    // Column major
    //dgemm("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);

    // row major - same as cblas_dgemm
    dgemm("N","N", &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);

    display(C, m*n);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k,
    B, n, beta, C, n);

    display(C, m*n);
}

void test_pdgemm() {
    /* Parameters */
    double   zero = 0.0e+0, one = 1.0e+0;
    MKL_INT i_zero = 0, i_one = 1, i_negone = -1;
    char trans = 'N';

    /*  Matrix descriptors */
    MDESC   descA, descB, descC, descU, descVt;

    /*  Local scalars */
    MKL_INT iam, nprocs, ictxt, myrow, mycol, nprow = 1, npcol = 2;
    MKL_INT n = 4, mb = 2, nb = 2, mp, nq, lld;
    MKL_INT i, j, info;

    /*  Local arrays */
    double  *A, *B, *C, *U, *S, *Vt;


    blacs_pinfo_( &iam, &nprocs );
    cout << "iam: " << iam << endl;
    cout << "nprocs: " << nprocs << endl;

    /*  Init workind 2D process grid */
    blacs_get_( &i_negone, &i_zero, &ictxt );
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    cout << "myrow: " << myrow << endl;
    cout << "mycol: " << mycol << endl;

    mp = numroc_( &n, &mb, &myrow, &i_zero, &nprow );
    nq = numroc_( &n, &nb, &mycol, &i_zero, &npcol );

    cout << "mp: " << mp << endl;
    cout << "nq: " << nq << endl;

    allocate(&A, (mp*nq));
    allocate(&B, (mp*nq));
    allocate(&C, (mp*nq));

    lld = MAX( mp, 1 );

    descinit_( descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
    descinit_( descB, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
    descinit_( descC, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );


    for (int i = 0; i < mp*nq; i++) {
        A[i] = 1;
    }

    for (int i = 0; i < mp*nq; i++) {
        B[i] = i;
    }

    for (int i = 0; i < mp*nq; i++) {
        C[i] = 0;
    }

    pdgemm_(&trans, &trans, &n, &n, &n, &one, A, &i_one, &i_one, descA, B, &i_one, &i_one, descB, &zero, C, &i_one, &i_one, descA);

    display(C, mp*nq);

    char jobu = 'V';
    char jobvt = 'V';
    double *work;
    double workpt;
    //MKL_INT lwork = 10*mp*nq; // if it is large enough to perform computation, then it would compute in the first try properly
    MKL_INT lwork = -1;

    allocate(&S, (mp));
    allocate(&U, (mp*nq));
    allocate(&Vt, (mp*nq));

    descinit_( descU, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
    descinit_( descVt, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );

    pdgesvd(&jobu, &jobvt, &n, &n, B, &i_one, &i_one, descA, S, U, &i_one, &i_one, descU, Vt, &i_one, &i_one, descVt, &workpt, &lwork, &info);

    cout << "workpt: " << workpt << endl;

    lwork = (int)workpt;
    work = (double*) calloc(lwork, sizeof( double ));

    pdgesvd(&jobu, &jobvt, &n, &n, B, &i_one, &i_one, descA, S, U, &i_one, &i_one, descU, Vt, &i_one, &i_one, descVt, work, &lwork, &info);

    cout << "info:" << info << endl;

    display(S, mp);
    display(U, mp*nq);
    display(Vt, mp*nq);
}

void test_dgesvd() {
    #define M 4
    #define N 4
    #define LDA M
    #define LDU M
    #define LDVT N

    int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
    double wkopt;
    double* work;
    /* Local arrays */
    double s[N], u[LDU*M], vt[LDVT*N];
    double a[LDA*N];
    for (int i = 0; i < LDA*N; i++) {
        a[i] = 1;
    }

    lwork = -1;
    cout << "wkopt: " << wkopt << endl;
    dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info );

    cout << "=================================" << endl;
    display(u, M*M);
    display(s, N);
    display(vt, N*N);

    cout << "wkopt: " << wkopt << endl;
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    /* Compute SVD */
    dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
     &info );
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }

    cout << "=================================" << endl;
    display(u, M*M);
    display(s, N);
    display(vt, N*N);
}
