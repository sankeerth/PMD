#include "../../common/headers/log.h"
#include "../headers/task.h"
#include "../headers/job.h"

void Task::create_task(Context& context, MPIContext& mpi_context) {
    switch(context.task) {
        case POD:
            Job::create_pod_job(context, mpi_context);
            break;

        case SparseCoding:
            Job::create_sparse_coding_job(context, mpi_context);
            break;

        default:
            LOGR("Wrong Task provided. Please provide task '0' for POD and '1' for Sparse Coding", mpi_context.my_rank, mpi_context.master);
            break;
    }
}
