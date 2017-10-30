#ifndef TEST_SCALAPACK_H
#define TEST_SCALAPACK_H

#include "mkl.h"
#include "mkl_scalapack.h"
#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "../common/headers/log.h"

typedef MKL_INT MDESC[ 9 ];

void test_dgemm();
void test_pdgemm();
void test_dgesvd();

#endif // TEST_SCALAPACK_H
