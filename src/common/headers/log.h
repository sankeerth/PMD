#ifndef LOG_H
#define LOG_H

#include <iostream>

using namespace std;

#ifdef LOGS_ON
#define NEWLINE cout << endl
#define LOG_(statement) cout << statement << " "
#define LOG(statement) cout << statement << endl
#define LOGD_(variable, value) cout << variable << ": " << value << " "
#define LOGD(variable, value) cout << variable << ": " << value << endl
#endif

#ifdef LOGS_RANK_ON
#define LOGR_(statement, my_rank, rank_to_display) if (my_rank == rank_to_display) cout << statement << " "
#define LOGR(statement, my_rank, rank_to_display) if (my_rank == rank_to_display) cout << statement << endl
#define LOGDR_(variable, value, my_rank, rank_to_display) if (my_rank == rank_to_display) cout << variable << ": " << value << " "
#define LOGDR(variable, value, my_rank, rank_to_display) if (my_rank == rank_to_display) cout << variable << ": " << value << endl
#endif

#ifdef SCALAPACK_LOG
#define NEWLINE cout << endl
#define SCALAPACK_LOG_(statement) cout << statement << " "
#define SCALAPACK_LOG(statement) cout << statement << endl
#define SCALAPACK_LOGD_(variable, value) cout << variable << ": " << value << " "
#define SCALAPACK_LOGD(variable, value) cout << variable << ": " << value << endl
#endif

#ifndef LOGS_ON
#define NEWLINE
#define LOG_(statement)
#define LOG(statement)
#define LOGD_(variable, value)
#define LOGD(variable, value)
#endif

#ifndef LOGS_RANK_ON
#define LOGR_(statement, my_rank, rank_to_display)
#define LOGR(statement, my_rank, rank_to_display)
#define LOGDR_(variable, value, my_rank, rank_to_display)
#define LOGDR(variable, value, my_rank, rank_to_display)
#endif

#ifndef SCALAPACK_LOG
#define NEWLINE
#define SCALAPACK_LOG_(statement)
#define SCALAPACK_LOG(statement)
#define SCALAPACK_LOGD_(variable, value)
#define SCALAPACK_LOGD(variable, value)
#endif

#endif // LOG_H
