#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::read_snapshots() {
    LOGR("=========== read_snapshots ===========", pod_context.my_rank, pod_context.master);

    LOGD_("rank", pod_context.my_rank);
    LOGD("start", pod_context.start_snapshot_num);

    LOG("=========== index_of_snapshot_filenames ===========");
    for (int i = 0; i < pod_context.index_of_snapshot_filenames.size(); i++) {
        LOGD_(i, pod_context.index_of_snapshot_filenames[i]);
    }
    NEWLINE;

    FILE *binfile;
    int details = 4;
    int constants = 4;
    size_t result;
    int *grid_details;
    float *constants_details;
    int imax_check;
    int jmax_check;
    int kmax_check;
    float xmach;
    float alpha;
    float reue;
    float time_instant;

    float *density;
    float *u_into_density;
    float *v_into_density;
    float *w_into_density;

    allocate(&grid_details, details);
    allocate(&constants_details, constants);

    allocate(&density, pod_context.grid_points_in_one_dim);
    allocate(&u_into_density, pod_context.grid_points_in_one_dim);
    allocate(&v_into_density, pod_context.grid_points_in_one_dim);
    allocate(&w_into_density, pod_context.grid_points_in_one_dim);

    //TODO: fix the allocate function for 3 dimensions
    //allocate(&pod_context.snapshots, 3 *pod_context. grid_points_in_one_dim, pod_context.num_snapshots);

    pod_context.snapshots = new float* [pod_context.snapshots_per_rank];
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        pod_context.snapshots[i] = new float[pod_context.grid_points_in_all_dim];
    }

    allocate(&pod_context.truncated_snapshots, pod_context.snapshots_per_rank * pod_context.truncated_grid_points_in_all_dim);

    // TODO: reading of files per process is assumed uniform distribution for now
    // Need to fix for non uniform distribution
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
        result = fread(u_into_density, sizeof(float), pod_context.grid_points_in_one_dim, binfile);
        result = fread(v_into_density, sizeof(float), pod_context.grid_points_in_one_dim, binfile);
        result = fread(w_into_density, sizeof(float), pod_context.grid_points_in_one_dim, binfile);

        for (unsigned long j = 0; j < pod_context.grid_points_in_one_dim; j++) {
            pod_context.snapshots[i][j] = u_into_density[j] / density[j];
            pod_context.snapshots[i][pod_context.grid_points_in_one_dim + j] = v_into_density[j] / density[j];
            pod_context.snapshots[i][2 * pod_context.grid_points_in_one_dim + j] = w_into_density[j] / density[j];
        }

        //TODO: A better way to truncate it than calling truncate 3 times for each of u,v and w velocity components
        truncate_snapshots(pod_context.snapshots[i], &pod_context.truncated_snapshots[i * pod_context.truncated_grid_points_in_all_dim], TupleI(pod_context.it_min, pod_context.it_max),
                       TupleI(pod_context.jt_min, pod_context.jt_max), TupleI(pod_context.kt_min, pod_context.kt_max),
                       Dimension(pod_context.imax, pod_context.jmax, pod_context.kmax));

        LOG("===========displaying truncated snapshots===========");
        //display(pod_context.truncated_snapshots[i], pod_context.nxt, pod_context.nyt, pod_context.nzt);
        //display(&pod_context.truncated_snapshots[i][truncated_grid_points], pod_context.nxt, pod_context.nyt, pod_context.nzt);
        //display(&pod_context.truncated_snapshots[i][2 * truncated_grid_points], pod_context.nxt, pod_context.nyt, pod_context.nzt);
        display(&pod_context.truncated_snapshots[i * pod_context.truncated_grid_points_in_all_dim], pod_context.truncated_grid_points_in_all_dim);
    }

    deallocate(&grid_details);
    deallocate(&constants_details);
    deallocate(&density);
    deallocate(&u_into_density);
    deallocate(&v_into_density);
    deallocate(&w_into_density);

    // freeing memory after use
    for (int i = 0; i < pod_context.snapshots_per_rank; i++) {
        delete [] pod_context.snapshots[i];
    }
    delete [] pod_context.snapshots;

    // Clearing the vector of file names as it is no longer required
    pod_context.filenames_to_distribute.clear();

    fclose(binfile);
}
