#include <fstream>
#include <iomanip>
#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::write_mean_flow_binary_1D_procs_along_col() {
    LOGR("=========== write_mean_flow_binary_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_output_directory();

    string str;
    str.append(pod_context.path_to_output_directory);
    str.append("mean_flow.b");

    FILE *binfile = fopen(str.c_str(), "wb");

    for (unsigned long i = 0; i < pod_context.truncated_grid_points_in_all_dim; i++) {
        fwrite(&pod_context.average[i], sizeof(float), 1, binfile);
    }

    fclose(binfile);
}

void POD::write_pod_modes_binary_1D_procs_along_col() {
    LOGR("=========== write_pod_modes_binary_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_output_directory();
    int files_to_write = MIN(pod_context.rank_eigen_values, pod_context.num_modes);

    for (int i = 0; (i < pod_context.index_of_snapshot_filenames.size()) && (pod_context.index_of_snapshot_filenames[i] < files_to_write); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_modes_bin-");
        str.append(patch::to_string(pod_context.index_of_snapshot_filenames[i]));
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < pod_context.truncated_grid_points_in_all_dim; j++) {
            fwrite(&pod_context.pod_bases[i * pod_context.truncated_grid_points_in_all_dim + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void POD::write_pod_coefficients_binary_1D_procs_along_col() {
    LOGR("=========== write_pod_coefficients_binary_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_output_directory();

    int coefficients_to_write = MIN(pod_context.num_modes, pod_context.rank_eigen_values);

    for (int i = 0, file_num = pod_context.my_rank; i < pod_context.index_of_snapshot_filenames.size(); i++, file_num += pod_context.num_procs) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_coefficients_bin-");
        str.append(patch::to_string(pod_context.total_num_solution_files + pod_context.start_index_of_snapshots + (file_num * pod_context.file_interval)),\
                   1, patch::to_string(pod_context.total_num_solution_files).length()-1);
        str.append(".b");
        FILE *binfile = fopen(str.c_str(), "wb");

        for (unsigned long j = 0; j < coefficients_to_write; j++) {
            fwrite(&pod_context.pod_coefficients[i * pod_context.num_snapshots + j], sizeof(float), 1, binfile);
        }

        fclose(binfile);
    }
}

void POD::write_pod_rms_error_binary_1D_procs_along_col() {
    LOGR("=========== write_pod_rms_error_binary_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_output_directory();

    for (int i = 0; i < pod_context.index_of_snapshot_filenames.size(); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_rms_error_bin-");
        str.append(patch::to_string(pod_context.index_of_snapshot_filenames[i]));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        fwrite(&pod_context.pod_rms_error[i], sizeof(float), 1, binfile);

        fclose(binfile);
    }
}

void POD::write_pod_modes_text_format_1D_procs_along_col() {
    LOGR("=========== write_pod_modes_text_format_1D_procs_along_col ===========", pod_context.my_rank, pod_context.master);

    // TODO: may be duplicate the function instead of complicating read_mesh
    read_mesh(pod_context.is_write_pod_modes_to_text_format);

    // create output directory if not present
    create_output_directory();
    int files_to_write = MIN(pod_context.rank_eigen_values, pod_context.num_modes);

    for (int i = 0; (i < pod_context.index_of_snapshot_filenames.size()) && (pod_context.index_of_snapshot_filenames[i] < files_to_write); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(pod_context.path_to_output_directory);
        str.append("pod_modes_txt.");
        str.append(patch::to_string(pod_context.index_of_snapshot_filenames[i]));
        ofstream fout(str);

        fout <<"  VARIABLES= \"X\",\"Y\",\"Z\",\"Mode_x\",\"Mode_y\",\"Mode_z\"\n";
        fout << "  ZONE T=" << "\"Mode 3D\"" << " I=\t" << pod_context.nxt << " J=\t" << pod_context.nyt << " K=\t" << pod_context.nzt << '\n';

        for (unsigned long j = 0; j < pod_context.truncated_grid_points; j++) {
            fout << "    " << setprecision(10) << pod_context.xt_grid[j] << "   " << setprecision(10) << pod_context.yt_grid[j] << "   " << setprecision(10) \
                 << pod_context.zt_grid[j] << "   " << setprecision(10) << pod_context.pod_bases[(i * pod_context.truncated_grid_points_in_all_dim) + j] << "   " << setprecision(10) \
                 << pod_context.pod_bases[(i * pod_context.truncated_grid_points_in_all_dim) + pod_context.truncated_grid_points + j] << "   " \
                 << setprecision(10) << pod_context.pod_bases[(i * pod_context.truncated_grid_points_in_all_dim) + (2 * pod_context.truncated_grid_points) + j] << "\n";
        }

        fout.close();
    }

    // free the memory here for each proc after the pod_modes are written.
    deallocate(&pod_context.xt_grid);
    deallocate(&pod_context.yt_grid);
    deallocate(&pod_context.zt_grid);
}
