#ifndef TASK_H
#define TASK_H

#include "../../common/headers/context.h"
#include "../../common/headers/mpi_context.h"

typedef enum {
    POD,
    SparseCoding
} TaskType;

class Task {
  public:
    static void create_task(Context& context, MPIContext& mpi_context);
};

#endif
