#include <stdlib.h>
#include "common/headers/log.h"
#include "common/headers/parser.h"
#include "common/headers/context.h"
#include "common/headers/mpi_context.h"
#include "controller/headers/task.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPIContext mpi_context;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_context.my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_context.num_procs);

    if (argc < 2) {
        LOGR("Not enough arguments", mpi_context.my_rank, mpi_context.master);
        LOGR("Usage mpiexec -np <num_procs> POD-<version>-release <input_file>", mpi_context.my_rank, mpi_context.master);
        MPI_Finalize();
        exit(1);
    }

    Context context;

    Parser parser;
    parser.parse_input_file(argv[1], context);

    if (!context.procs_along_row && mpi_context.num_procs > context.num_snapshots) {
        LOGR("More procs used than num of snapshots or num of modes. Execute again with num of procs less than or equal to num of snapshots", mpi_context.my_rank, mpi_context.master);
        MPI_Finalize();
        exit(1);
    }

    Task::create_task(context, mpi_context);

    MPI_Finalize();
    return 0;
}
