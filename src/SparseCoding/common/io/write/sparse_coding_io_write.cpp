#include "../../../../common/headers/log.h"
#include "../../../sparse_coding.h"

void SparseCoding::write_sparse_coding_rms_error_binary() {
    LOGR("=========== write_sparse_coding_rms_error_binary ===========", sparse_context.my_rank, sparse_context.master);

    // create output directory if not present
    create_directory(sparse_context.path_to_output_directory);

    int files_to_write = MIN(sparse_context.rank_eigen_values, sparse_context.num_modes);

    for (int i = 0; (i < sparse_context.index_of_snapshot_filenames.size()) && (sparse_context.index_of_snapshot_filenames[i] < files_to_write); i++) {
        // TODO: May be modify the substring instead of creating a new one each time
        string str;
        str.append(sparse_context.path_to_output_directory);
        str.append("sparse_coding_rms_error_bin-");
        str.append(patch::to_string(sparse_context.index_of_snapshot_filenames[i]));
        str.append(".b");

        FILE *binfile = fopen(str.c_str(), "wb");

        fwrite(&sparse_context.sparse_coding_rms_error[i], sizeof(float), 1, binfile);

        fclose(binfile);
    }
}
