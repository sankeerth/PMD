#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <sstream>
#include <vector>

#define MACHINE_PRECISION 1e-15

extern "C" {
    int numroc_(int*, int*, int*, int*, int*);
    int descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
}

enum axis {
    i, j, k
};

class TupleI {
  public:
    TupleI(int s, int e) {
        start = s;
        end = e;
    }

    int start;
    int end;
};

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

class Dimension {
  public:
    Dimension(int x, int y, int z, int dim = 3) {
        x_axis_elem = x;
        y_axis_elem = y;
        z_axis_elem = z;
        dimensions = dim;
    }

    int x_axis_elem;
    int y_axis_elem;
    int z_axis_elem;
    int dimensions;
};

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

off_t fsize(const char *filename);
void measure_time_for_function(double& start_time, std::string function_name, int& my_rank, int& master);
int cmpfunc (const void * a, const void * b);
void create_file_list(std::string file_name, int& start, int &end, int& interval, std::string& path, std::string& prefix, int total_files, std::string& extension);
void create_directory(std::string path);
void allocate(int **array, int size);
void allocate(float **array, int size);
void allocate(double **array, int size);
void deallocate(int **array);
void deallocate(float **array);
void deallocate(double **array);
void initialize(int *array, int size);
void initialize(float *array, int size);
void initialize(double *array, int size);
void display(float *array, int length);
void display(int *array, int length);
void display(double *array, int length);
void display(float *array, int i, int j, int k);
void truncate_array(float *array, float *truncated_array, TupleI x_tuple, TupleI y_tuple, TupleI z_tuple, Dimension dim);
void truncate_snapshots(float *array, float *truncated_array, TupleI x_tuple, TupleI y_tuple, TupleI z_tuple, Dimension dim);
void duplicate(int *array, int *duplicate, int size);
void duplicate(float *array, float *duplicate, int size);
void duplicate(double *array, double *duplicate, int size);
void duplicate(double *array, float *duplicate, int size);
void duplicate(float *array, double *duplicate, int size);
float norm(float *array, int length);
double norm(double *array, int length);
void normalize(float *array, int length);
void normalize(double *array, int length);
void get_max_and_pos(float *array, int size, float *max, int *pos);
void get_abs_max_and_pos(float *array, int size, float *max, int *pos);
int get_num_elements_greater_than_threshold_along_row(float *array, int rows, int cols, int row, float threshold);
int get_num_elements_greater_than_threshold_along_col(float *array, int rows, int col, float threshold);
std::vector<int> get_indices_nonzero_each_row(int row, float *array, int rows, int cols);
std::vector<int> get_indices_nonzero_each_col(float *array, int rows, int col);
void round_off_below_machine_precision_to_zero(float *array, int length);
void round_off_below_diff_max_and_threshold_to_zero(float *array, int length, float max, float threshold);

#endif // UTILITIES_H
