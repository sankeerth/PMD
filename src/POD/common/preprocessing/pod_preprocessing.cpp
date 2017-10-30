#include "../../../common/headers/log.h"
#include "../../pod.h"

void POD::verify_grid(int imax_check, int jmax_check, int kmax_check) {
    LOGR("=========== verify_grid ===========", pod_context.my_rank, pod_context.master);

    if (pod_context.imax != imax_check || pod_context.jmax != jmax_check || pod_context.kmax != kmax_check) {
        LOG("Grid size does not match the user input given in input_file");
        MPI_Finalize();
        exit(0);
    }
}

void POD::read_mesh(bool write_pod_modes) {
    LOGR("=========== read_mesh ===========", pod_context.my_rank, pod_context.master);

    int details = 4;
    unsigned long num_records=fsize(pod_context.mesh_file.c_str()) / 4;
    size_t result;
    int *grid_details;
    float *record;

    LOGD("num records", num_records);
    FILE* binfile = fopen(pod_context.mesh_file.c_str(),"rb") ;
    //TODO: change the value to dynamic after reading plots
    allocate(&grid_details, details);
    result = fread(grid_details, sizeof(int), details, binfile);

    pod_context.num_plots = grid_details[0];
    int imax_check = grid_details[1];
    int jmax_check = grid_details[2];
    int kmax_check = grid_details[3];

    // check if the user provided grid details are same as one read from grid, else exit!
    verify_grid(imax_check, jmax_check, kmax_check);

    display(grid_details, details);

    allocate(&record, pod_context.grid_points_in_one_dim);

    for (int j = 0; j < pod_context.dimensions; j++) {
        result = fread(record, sizeof(float), pod_context.grid_points_in_one_dim, binfile) ;
        // TODO: since truncation of grid is skipped for efficiency but it is needed to write modes,
        // this should be moved somewhere else which can be executed only if write modes is true
        if (write_pod_modes) {
            compute_truncated_grid(record, static_cast<axis> (j));
        } else {
            make_grid_with_averaging_points(record, static_cast<axis> (j));
        }
    }

    deallocate(&record);
    fclose(binfile);
}

void POD::compute_truncated_grid(float *record, axis ax) {
    LOGR("=========== compute_truncated_grid ===========", pod_context.my_rank, pod_context.master);

    switch (ax) {
    case i:
        allocate(&pod_context.xt_grid, pod_context.nxt * pod_context.nyt * pod_context.nzt);
        truncate_array(record, pod_context.xt_grid, TupleI(pod_context.it_min-1, pod_context.it_max-1),
                       TupleI(pod_context.jt_min-1, pod_context.jt_max-1), TupleI(pod_context.kt_min-1, pod_context.kt_max-1),
                       Dimension((pod_context.imax), (pod_context.jmax), (pod_context.kmax)));
        break;
    case j:
        allocate(&pod_context.yt_grid, pod_context.nxt * pod_context.nyt * pod_context.nzt);
        truncate_array(record, pod_context.yt_grid, TupleI(pod_context.it_min-1, pod_context.it_max-1),
                       TupleI(pod_context.jt_min-1, pod_context.jt_max-1), TupleI(pod_context.kt_min-1, pod_context.kt_max-1),
                       Dimension((pod_context.imax), (pod_context.jmax), (pod_context.kmax)));
        break;
    case k:
        allocate(&pod_context.zt_grid, pod_context.nxt * pod_context.nyt * pod_context.nzt);
        truncate_array(record, pod_context.zt_grid, TupleI(pod_context.it_min-1, pod_context.it_max-1),
                       TupleI(pod_context.jt_min-1, pod_context.jt_max-1), TupleI(pod_context.kt_min-1, pod_context.kt_max-1),
                       Dimension((pod_context.imax), (pod_context.jmax), (pod_context.kmax)));
        break;
    }
}

void POD::make_grid_with_averaging_points(float *grid, axis ax) {
    LOGR("=========== make_grid_with_averaging_points ===========", pod_context.my_rank, pod_context.master);

    unsigned long z_stride, y_stride, y_stride_pre, z_stride_pre;
    unsigned long x_grid_index = 0, y_grid_index = 0, z_grid_index = 0;
    switch(ax) {
        case i:
            LOG("===========computing x_grid===========");
            allocate(&pod_context.x_grid, (pod_context.imax-1)*pod_context.jmax*pod_context.kmax);
            for (int z = 0; z < pod_context.kmax; z++) {
                z_stride = z * pod_context.jmax * pod_context.imax;
                for (int y = 0; y < pod_context.jmax; y++) {
                    y_stride = y * pod_context.imax;
                    for (int x = 1; x < pod_context.imax; x++) {
                        //cout << (x + y_stride + z_stride) + (x + y_stride + z_stride) + 1 << " ";
                        pod_context.x_grid[x_grid_index] = (grid[(x + y_stride + z_stride)] + grid[(x-1 + y_stride + z_stride)])/2;
                        //cout << x_grid_index << ": "<<  pod_context.x_grid[x_grid_index] << " ";
                        x_grid_index += 1;
                    }
                }
            }
            display(pod_context.x_grid, pod_context.imax-1, pod_context.jmax, pod_context.kmax);
        break;

        case j:
            LOG("===========computing y_grid===========");
            allocate(&pod_context.y_grid, pod_context.imax*(pod_context.jmax-1)*pod_context.kmax);
            for (int z = 0; z < pod_context.kmax; z++) {
                z_stride = z * pod_context.jmax * pod_context.imax;
                for (int y = 1; y < pod_context.jmax; y++) {
                    y_stride_pre = (y-1) * pod_context.imax;
                    y_stride = y * pod_context.imax;
                    for (int x = 0; x < pod_context.imax; x++) {
                        //cout << (x + y_stride_pre + z_stride) + (x + y_stride + z_stride) + 2 << " ";
                        pod_context.y_grid[y_grid_index] = (grid[(x + y_stride_pre + z_stride)] + grid[(x + y_stride + z_stride)])/2;
                        //cout << y_grid_index << ": " << pod_context.y_grid[y_grid_index] << " ";
                        y_grid_index += 1;
                    }
                }
            }
            display(pod_context.y_grid, pod_context.imax, pod_context.jmax-1, pod_context.kmax);
        break;

        case k:
            LOG("===========computing z_grid===========");
            allocate(&pod_context.z_grid, pod_context.imax*pod_context.jmax*(pod_context.kmax-1));
            for (int z = 1; z < pod_context.kmax; z++) {
                z_stride_pre = (z-1) * pod_context.jmax * pod_context.imax;
                z_stride = z * pod_context.jmax * pod_context.imax;
                for (int y = 0; y < pod_context.jmax; y++) {
                    y_stride = y * pod_context.imax;
                    for (int x = 0; x < pod_context.imax; x++) {
                        //cout << (x + y_stride + z_stride_pre) + (x + y_stride + z_stride) + 2 << " ";
                        pod_context.z_grid[z_grid_index] = (grid[(x + y_stride + z_stride_pre)] + grid[(x + y_stride + z_stride)])/2;
                        //cout << z_grid_index << ": " << pod_context.z_grid[z_grid_index] << " ";
                        z_grid_index += 1;
                    }
                }
            }
            display(pod_context.z_grid, pod_context.imax, pod_context.jmax, pod_context.kmax-1);
        break;
    }
}

void POD::compute_differential_of_grid(axis ax) {
    LOGR("=========== compute_differential_of_grid ===========", pod_context.my_rank, pod_context.master);

    unsigned long z_stride, y_stride, y_stride_pre, z_stride_pre;
    unsigned long dxc_grid_index = 0, dyc_grid_index = 0, dzc_grid_index = 0;
    switch(ax) {
        case i:
            LOG("===========computing dxc_grid===========");
            allocate(&pod_context.dxc_grid, (pod_context.imax-2)*(pod_context.jmax-2)*(pod_context.kmax-2));
            for (int z = 0; z < pod_context.kmax-2; z++) {
                z_stride = z * pod_context.jmax * (pod_context.imax-1);
                for (int y = 0; y < pod_context.jmax-2; y++) {
                    y_stride = y * (pod_context.imax-1);
                    for (int x = 1; x < pod_context.imax-1; x++) {
                        pod_context.dxc_grid[dxc_grid_index] = (pod_context.x_grid[x + y_stride + z_stride] - pod_context.x_grid[x - 1 + y_stride + z_stride]);
                        dxc_grid_index++;
                    }
                }
            }
            display(pod_context.dxc_grid, pod_context.imax-2, pod_context.jmax-2, pod_context.kmax-2);
        break;

        case j:
            LOG("===========computing dyc_grid===========");
            allocate(&pod_context.dyc_grid, (pod_context.imax-2)*(pod_context.jmax-2)*(pod_context.kmax-2));
            for (int z = 0; z < pod_context.kmax-2; z++) {
                z_stride = z * (pod_context.jmax-1) * pod_context.imax;
                for (int y = 1; y < pod_context.jmax-1; y++) {
                    y_stride_pre = (y-1) * pod_context.imax;
                    y_stride = y * pod_context.imax;
                    for (int x = 0; x < pod_context.imax-2; x++) {
                        pod_context.dyc_grid[dyc_grid_index] = (pod_context.y_grid[x + y_stride + z_stride] - pod_context.y_grid[x + y_stride_pre + z_stride]);
                        dyc_grid_index++;
                    }
                }
            }
            display(pod_context.dyc_grid, pod_context.imax-2, pod_context.jmax-2, pod_context.kmax-2);
        break;

        case k:
            LOG("===========computing dzc_grid===========");
            allocate(&pod_context.dzc_grid, (pod_context.imax-2)*(pod_context.jmax-2)*(pod_context.kmax-2));
            for (int z = 1; z < pod_context.kmax-1; z++) {
                z_stride_pre = (z-1) * pod_context.jmax * pod_context.imax;
                z_stride = z * pod_context.jmax * pod_context.imax;
                for (int y = 0; y < pod_context.jmax-2; y++) {
                    y_stride = y * pod_context.imax;
                    for (int x = 0; x < pod_context.imax-2; x++) {
                        pod_context.dzc_grid[dzc_grid_index] = (pod_context.z_grid[x + y_stride + z_stride] - pod_context.z_grid[x + y_stride + z_stride_pre]);
                        dzc_grid_index++;
                    }
                }
            }
            display(pod_context.dzc_grid, pod_context.imax-2, pod_context.jmax-2, pod_context.kmax-2);
        break;
    }
}

void POD::grid_metrics() {
    LOGR("=========== grid_metrics ===========", pod_context.my_rank, pod_context.master);

    compute_differential_of_grid(axis::i);
    compute_differential_of_grid(axis::j);
    compute_differential_of_grid(axis::k);

    // freeing memory after use
    deallocate(&pod_context.x_grid);
    deallocate(&pod_context.y_grid);
    deallocate(&pod_context.z_grid);
}

void POD::compute_mesh_volume() {
    LOGR("=========== compute_mesh_volume ===========", pod_context.my_rank, pod_context.master);

    allocate(&pod_context.vol_grid, (pod_context.imax-2) * (pod_context.jmax-2) * (pod_context.kmax-2));
    allocate(&pod_context.truncated_vol_grid, pod_context.nxt * pod_context.nyt * pod_context.nzt);
    unsigned long z_stride, y_stride;
    unsigned long vol_grid_index = 0;
    for (int z = 0; z < pod_context.kmax-2; z++) {
        z_stride = z * (pod_context.jmax-2) * (pod_context.imax-2);
        for (int y = 0; y < pod_context.jmax-2; y++) {
            y_stride = y * (pod_context.imax-2);
            for (int x = 0; x < pod_context.imax-2; x++) {
                pod_context.vol_grid[vol_grid_index] = (pod_context.dxc_grid[x + y_stride + z_stride]) * (pod_context.dyc_grid[x + y_stride + z_stride]) * (pod_context.dzc_grid[x + y_stride + z_stride]);
                vol_grid_index++;
            }
        }
    }
    display(pod_context.vol_grid, (pod_context.imax-2), (pod_context.jmax-2), (pod_context.kmax-2));

    truncate_array(pod_context.vol_grid, pod_context.truncated_vol_grid, TupleI(pod_context.it_min-1, pod_context.it_max-1),
                   TupleI(pod_context.jt_min-1, pod_context.jt_max-1), TupleI(pod_context.kt_min-1, pod_context.kt_max-1),
                   Dimension((pod_context.imax-2), (pod_context.jmax-2), (pod_context.kmax-2)));

    LOG("===========vol_grid after truncating===========");
    display(pod_context.truncated_vol_grid, pod_context.nxt, pod_context.nyt, pod_context.nzt);

    // freeing memory after use
    deallocate(&pod_context.vol_grid);

    deallocate(&pod_context.dxc_grid);
    deallocate(&pod_context.dyc_grid);
    deallocate(&pod_context.dzc_grid);
}

void POD::distribute_truncated_vol_grid() {
    LOGR("=========== distribute_truncated_vol_grid ===========", pod_context.my_rank, pod_context.master);

    // already allocated in master in compute_mesh_volume. allocating in other procs and distributing it
    if (pod_context.my_rank != pod_context.master) {
        allocate(&pod_context.truncated_vol_grid, pod_context.truncated_grid_points);
    }

    MPI_Bcast(pod_context.truncated_vol_grid, pod_context.truncated_grid_points, MPI_FLOAT, pod_context.master, MPI_COMM_WORLD);
}
