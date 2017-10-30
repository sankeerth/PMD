#ifndef SCALAPACK_HELPERS_H
#define SCALAPACK_HELPERS_H

#include "mkl.h"
#include "mkl_scalapack.h"
#include "mkl_blacs.h"
#include "mkl_pblas.h"

typedef MKL_INT MDESC[ 9 ];

void matrix_mul_1D_process_grid(float *A, float *B, float **C, int& rows_C_local, int& cols_C_local, int m, int k, int n,\
                                char transpose_A, char transpose_B, int nprow, int npcol, int mb_A, int nb_A, int mb_B, int nb_B, bool is_A_same_as_B);
void singular_value_decomposition(float *A, char jobu, char jobvt, int m, int n, float **eigen_values, float **U, float **Vt, int &size_of_eigen_values, int& rows_U_local, int& cols_U_local, \
                                  int& rows_Vt_local, int& cols_Vt_local, int nprow, int npcol, int mb_A, int nb_A, int mb_U, int nb_U, int mb_Vt, int nb_Vt);
void matrix_transpose(float *A, int m, int n, float **C, int &rows_C_local, int &cols_C_local, int nprow, int npcol, int mb_A, int nb_A, int mb_C, int nb_C);
void matrix_vector_mul(float *A, int m, int n, float *X, float **Y, int& out_vector_length, char transpose_A, int mb_A, int nb_A, int nprow, int npcol,\
                       int ia, int ja, int ix, int jx, int iy, int jy, int incx, int incy);

void matrix_mul_1D_process_grid(double *A, double *B, double **C, int& rows_C_local, int& cols_C_local, int m, int k, int n,\
                                char transpose_A, char transpose_B, int nprow, int npcol, int mb_A, int nb_A, int mb_B, int nb_B, bool is_A_same_as_B);
void matrix_vector_mul(double *A, int m, int n, double *X, double **Y, int& out_vector_length, char transpose_A, int mb_A, int nb_A, int nprow, int npcol,\
                       int ia, int ja, int ix, int jx, int iy, int jy, int incx, int incy);


#endif // SCALAPACK_HELPERS_H
