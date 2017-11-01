#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::read_truncated_snapshots_1D_procs_along_col() {
    LOGR("=========== read_truncated_snapshots_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

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

    allocate(&pod_context.truncated_snapshots, pod_context.snapshots_per_rank * pod_context.truncated_grid_points_in_all_dim);

    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        LOGD_("rank", pod_context.my_rank);
        LOGD("reading file", pod_context.filenames_to_distribute[pod_context.index_of_snapshot_filenames[i]]);
        binfile = fopen(pod_context.filenames_to_distribute[pod_context.index_of_snapshot_filenames[i]].c_str(),"rb") ;

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

        for (unsigned long  current_index = 0; current_index < pod_context.truncated_grid_points_in_all_dim; current_index++) {
            offset_index = (current_index / pod_context.truncated_grid_points);
            density_index = pod_context.truncated_snapshot_indices[current_index % pod_context.truncated_grid_points];
            velocity_index = offset_index * pod_context.grid_points_in_one_dim + density_index;

            pod_context.truncated_snapshots[i * pod_context.truncated_grid_points_in_all_dim + current_index] = velocity_into_density[velocity_index] / density[density_index];
        }

        fclose(binfile);
    }

    LOG("=========== truncated_snapshots ===========");
    display(pod_context.truncated_snapshots, pod_context.snapshots_per_rank * pod_context.truncated_grid_points_in_all_dim);

    deallocate(&grid_details);
    deallocate(&constants_details);
    deallocate(&density);
    deallocate(&velocity_into_density);
    deallocate(&pod_context.truncated_snapshot_indices);
}

void POD::read_mean_flow_binary_1D_procs_along_col() {
    LOGR("=========== read_mean_flow_binary_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    size_t result;
    string str;
    str.append(pod_context.path_to_output_directory);
    str.append("mean_flow.b");
    size_t num_mean_flow = fsize(str.c_str()) / 4;
    FILE *binfile = fopen(str.c_str(), "rb");

    allocate(&pod_context.average, num_mean_flow);
    result = fread(pod_context.average, sizeof(float), num_mean_flow, binfile);
    fclose(binfile);

    display(pod_context.average, num_mean_flow);
}

void POD::read_pod_modes_1D_procs_along_col() {
    LOGR("=========== read_pod_modes_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    size_t result;
    float * buffer = NULL;
    allocate(&pod_context.pod_bases, 3 * pod_context.truncated_grid_points * pod_context.num_modes_in_my_rank);
    // TODO: optimize to read directly to pod_bases than to buffer and then copy it. This will save memory
    allocate(&buffer, 3 * pod_context.truncated_grid_points);

    for (int i = 0; i < pod_context.num_modes_in_my_rank; i++) {
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_modes_bin-");
        str.append(patch::to_string(pod_context.index_of_snapshot_filenames[i]));
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "rb");
        size_t num_pod_modes = fsize(str.c_str()) / 4;

        // size of pod_bases has be the same as the truncated snapshots
        if (3 * pod_context.truncated_grid_points != num_pod_modes) {
            LOG("Number of points in single pod_bases file is not equal to 3 * truncated grid points");
            LOG("There is an error in mostly file creation of pod_bases_bin file.");
            MPI_Finalize();
            exit(0);
        }

        result = fread(buffer, sizeof(float), num_pod_modes, binfile);
        for (unsigned long j = 0; j < num_pod_modes; j++) {
            pod_context.pod_bases[i * num_pod_modes + j] = buffer[j];
        }

        fclose(binfile);
    }

    deallocate(&buffer);
    display(pod_context.pod_bases, 3 * pod_context.truncated_grid_points * pod_context.snapshots_per_rank);
}
