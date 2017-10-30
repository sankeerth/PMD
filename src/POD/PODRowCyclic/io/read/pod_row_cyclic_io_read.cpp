#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::read_truncated_snapshots_1D_procs_along_row() {
    LOGR("=========== compute_average_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    FILE *binfile;
    int details = 4;
    int constants = 4;
    size_t result;
    int *grid_details = NULL;
    float *constants_details = NULL, *velocity_into_density = NULL, *density = NULL;
    int imax_check;
    int jmax_check;
    int kmax_check;
    float xmach, alpha, reue, time_instant;

    allocate(&grid_details, details);
    allocate(&constants_details, constants);

    allocate(&density, pod_context.grid_points_in_one_dim);
    allocate(&velocity_into_density, pod_context.grid_points_in_all_dim);

    allocate(&pod_context.truncated_snapshots, pod_context.num_snapshots * pod_context.num_snapshot_points_per_proc);

    for (int i = 0; i < pod_context.num_snapshots; i++) {
        LOGD_("rank", pod_context.my_rank);
        LOGD("reading file", pod_context.filenames_to_distribute[i]);
        binfile = fopen(pod_context.filenames_to_distribute[i].c_str(),"rb") ;

        //TODO: change the value to dynamic after reading plots
        result = fread(grid_details, sizeof(int), details, binfile);

        pod_context.num_plots = grid_details[0];
        imax_check = grid_details[1];
        jmax_check = grid_details[2];
        kmax_check = grid_details[3];

        result = fread(constants_details, sizeof(float), constants, binfile);

        xmach = constants_details[0];
        alpha = constants_details[1];
        reue = constants_details[2];
        time_instant = constants_details[3];

        // check if the user provided grid details are same as one read from grid, else exit!
        verify_grid(imax_check, jmax_check, kmax_check);

        display(grid_details, details);
        display(constants_details, constants);

        result = fread(density, sizeof(float), pod_context.grid_points_in_one_dim, binfile);
        result = fread(velocity_into_density, sizeof(float), pod_context.grid_points_in_all_dim, binfile);

        unsigned long offset_index, velocity_index, density_index;

        for (unsigned long j = 0, current_index = pod_context.my_rank; j < pod_context.num_snapshot_points_per_proc; j++, current_index += pod_context.num_procs) {
            offset_index = (current_index / pod_context.truncated_grid_points);
            density_index = pod_context.truncated_snapshot_indices[current_index % pod_context.truncated_grid_points];
            velocity_index = offset_index * pod_context.grid_points_in_one_dim + density_index;

            //LOGD_("off", offset_index);
            //LOGD_("den", density_index);
            //LOGD("vel", velocity_index);

            pod_context.truncated_snapshots[i * pod_context.num_snapshot_points_per_proc + j] = velocity_into_density[velocity_index] / density[density_index];
        }
        NEWLINE;
        fclose(binfile);
    }

    LOG("=========== truncated_snapshots ===========");
    display(pod_context.truncated_snapshots, pod_context.num_snapshots * pod_context.num_snapshot_points_per_proc);

    deallocate(&grid_details);
    deallocate(&constants_details);
    deallocate(&density);
    deallocate(&velocity_into_density);
    deallocate(&pod_context.truncated_snapshot_indices);
}
