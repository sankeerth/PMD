#include "../../../../common/headers/log.h"
#include "../../../pod.h"

void POD::write_eigen_values_binary() {
    LOGR("=========== write_eigen_values_binary ===========", pod_context.my_rank, pod_context.master);

    // create output directory if not present
    create_output_directory();

    string str;
    str.append(pod_context.path_to_output_directory);
    str.append("eigen_values_bin.b");

    FILE *binfile = fopen(str.c_str(), "wb");

    for (unsigned long i = 0; i < pod_context.rank_eigen_values; i++) {
        fwrite(&pod_context.eigen_values[i], sizeof(float), 1, binfile);
    }

    fclose(binfile);
}
