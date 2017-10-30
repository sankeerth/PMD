#include<iostream>
#include <fstream>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../headers/log.h"
#include "../headers/utilities.h"

using namespace std;

/**
    Returns the size of the file in bytes

    @param filename Name of the file
    @return void
*/
off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1;
}

void measure_time_for_function(double& start_time, string function_name, int& my_rank, int& master) {
    double end_time;

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == master) {
        end_time = MPI_Wtime();
        cout << "Time taken until function " << function_name << ": " << (end_time-start_time) << endl;
    }
}

int cmpfunc (const void * a, const void * b)
{
    int f = *((float*)a);
    int s = *((float*)b);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

/**
    Compares two given values

    @param a value to compare
    @param b value to compare
    @return void
*/
void create_file_list(string file_name, int& start, int& end, int& interval, string& path, string& prefix, int total_files, string& extension) {
    ofstream file(file_name.c_str());

    for (int i = 0, file_num = start; i < end; ++i, file_num += interval) {
        string str;
        str.append(path);
        str.append(prefix);
        str.append(patch::to_string(total_files + file_num), 1, patch::to_string(total_files).length()-1);
        if (extension != "0") {
            str.append(extension);
        }

        file << str << endl;
    }
    file.close();
}

void create_directory(string path) {
    struct stat st = {0};

    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0700);
    }
}

void allocate(int **array, int size) {
    *array = (int*) malloc (size * sizeof(int));
    if (!(*array)) {
        cout << "FAILED TO ALLOCATE MEMORY" << endl;
    }
}

void allocate(float **array, int size) {
    *array = (float*) malloc (size * sizeof(float));
    if (!(*array)) {
        cout << "FAILED TO ALLOCATE MEMORY" << endl;
    }
}

void allocate(double **array, int size) {
    *array = (double*) malloc (size * sizeof(double));
    if (!(*array)) {
        cout << "FAILED TO ALLOCATE MEMORY" << endl;
    }
}

void deallocate(int **array) {
    if (*array != NULL) {
        free(*array);
        *array = NULL;
    }
}

void deallocate(float **array) {
    if (*array != NULL) {
        free(*array);
        *array = NULL;
    }
}

void deallocate(double **array) {
    if (*array != NULL) {
        free(*array);
        *array = NULL;
    }
}

void initialize(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0;
    }
}

void initialize(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0;
    }
}

void initialize(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0;
    }
}

#ifdef DISPLAY_ON
void display(float *array, int length) {
    for (int i = 0; i < length; i++) {
        cout << i << ":" << array[i] << " ";
    }
    cout << endl;
}

void display(int *array, int length) {
    for (int i = 0; i < length; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

void display(double *array, int length) {
    for (int i = 0; i < length; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

void display(float *array, int i, int j, int k) {
    int z_stride, y_stride;
    for (int z = 0; z < k; z++) {
        z_stride = z * j * i;
        for (int y = 0; y < j; y++) {
            y_stride = y * i;
            for (int x = 0; x < i; x++) {
                cout << x + y_stride + z_stride << ": " << array[x + y_stride + z_stride] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}
#endif

#ifndef DISPLAY_ON
void display(float *array, int length) {}
void display(int *array, int length) {}
void display(double *array, int length) {}
void display(float *array, int i, int j, int k) {}
#endif

void truncate_array(float *array, float *truncated_array, TupleI x_tuple, TupleI y_tuple, TupleI z_tuple, Dimension dim) {
    int z_stride, y_stride;
    int truncated_array_index = 0;
    for (int z = z_tuple.start; z <= z_tuple.end; z++) {
        z_stride = z * dim.x_axis_elem * dim.y_axis_elem;
        for (int y = y_tuple.start; y <= y_tuple.end; y++) {
            y_stride = y * dim.x_axis_elem;
            for (int x = x_tuple.start; x <= x_tuple.end; x++) {
                truncated_array[truncated_array_index] = array[x + y_stride + z_stride];
                truncated_array_index++;
            }
        }
    }
}

void truncate_snapshots(float *array, float *truncated_array, TupleI x_tuple, TupleI y_tuple, TupleI z_tuple, Dimension dim) {
    int z_stride, y_stride;
    int truncated_array_index = 0;
    int grid_points_in_one_dimension = dim.x_axis_elem * dim.y_axis_elem * dim.z_axis_elem;
    for (int i = 0; i < dim.dimensions; i++) {
        for (int z = z_tuple.start; z <= z_tuple.end; z++) {
            z_stride = z * dim.x_axis_elem * dim.y_axis_elem;
            for (int y = y_tuple.start; y <= y_tuple.end; y++) {
                y_stride = y * dim.x_axis_elem;
                for (int x = x_tuple.start; x <= x_tuple.end; x++) {
                    truncated_array[truncated_array_index] = (array+i*grid_points_in_one_dimension)[x + y_stride + z_stride];
                    truncated_array_index++;
                }
            }
        }
    }
}

void duplicate(int *array, int *duplicate, int size) {
    for (int i = 0; i < size; i++) {
        duplicate[i] = array[i];
    }
}

void duplicate(float *array, float *duplicate, int size) {
    for (int i = 0; i < size; i++) {
        duplicate[i] = array[i];
    }
}

void duplicate(double *array, double *duplicate, int size) {
    for (int i = 0; i < size; i++) {
        duplicate[i] = array[i];
    }
}

void duplicate(double *array, float *duplicate, int size) {
    for (int i = 0; i < size; i++) {
        duplicate[i] = (float) array[i];
    }
}

void duplicate(float *array, double *duplicate, int size) {
    for (int i = 0; i < size; i++) {
        duplicate[i] = (double) array[i];
    }
}

float norm(float *array, int length) {
    double norm_val = 0;
    for (int i = 0; i < length; i++) {
        norm_val += (double) array[i] * (double) array[i];
    }

    norm_val = sqrt(norm_val);
    return (float) norm_val;
}

double norm(double *array, int length) {
    double norm_val = 0;
    for (int i = 0; i < length; i++) {
        norm_val += array[i] * array[i];
    }

    norm_val = sqrt(norm_val);
    return norm_val;
}

void normalize(float *array, int length) {
    float norm_val = norm(array, length);

    for (int i = 0; i < length; i++) {
        array[i] = array[i] / norm_val;
    }
}

void normalize(double *array, int length) {
    double norm_val = norm(array, length);

    for (int i = 0; i < length; i++) {
        array[i] = array[i] / norm_val;
    }
}

void get_max_and_pos(float *array, int size, float *max, int *pos) {
    float max_val = array[0];
    *max = array[0];
    *pos = 0;

    for (int i = 0; i < size; i++) {
        max_val = MAX(max_val, array[i]);
        if (max_val > *max) {
            *max = max_val;
            *pos = i;
        }
    }
}

void get_abs_max_and_pos(float *array, int size, float *max, int *pos) {
    float max_val = 0;
    *max = 0;
    *pos = 0;

    for (int i = 0; i < size; i++) {
        max_val = MAX(max_val, fabs(array[i]));
        if (max_val > *max) {
            *max = max_val;
            *pos = i;
        }
    }
}

int get_num_elements_greater_than_threshold_along_row(float *array, int rows, int cols, int row, float threshold) {
    int count = 0;
    for (int i = 0; i < cols; i++) {
        if (fabs(array[i * rows + row]) > threshold) {
            count += 1;
        }
    }
    return count;
}

int get_num_elements_greater_than_threshold_along_col(float *array, int rows, int col, float threshold) {
    int count = 0;
    for (int i = 0; i < rows; i++) {
        if (fabs(array[col * rows + i]) > threshold) {
            count += 1;
        }
    }
    return count;
}

vector<int> get_indices_nonzero_each_row(int row, float *array, int rows, int cols) {
    // TODO: send relevant_data_indices as a parameter to avoid copying the vector when returned
    vector<int> relevant_data_indices;
    for (int i = 0; i < cols; i++) {
        if (array[i * rows + row] != 0) {
            relevant_data_indices.push_back(i);
        }
    }
    return relevant_data_indices;
}

vector<int> get_indices_nonzero_each_col(float *array, int rows, int col) {
    // TODO: send relevant_data_indices as a parameter to avoid copying the vector when returned
    vector<int> relevant_data_indices;
    for (int i = 0; i < rows; i++) {
        if (array[col * rows + i] != 0) {
            relevant_data_indices.push_back(i);
        }
    }
    return relevant_data_indices;
}

void round_off_below_machine_precision_to_zero(float *array, int length) {
    for (int i = 0; i < length; i++) {
        if (fabs(array[i]) < MACHINE_PRECISION) {
            array[i] = 0;
        }
    }
}

void round_off_below_diff_max_and_threshold_to_zero(float *array, int length, float max, float threshold) {
    for (int i = 0; i < length; i++) {
        if (fabs(array[i])/(fabs(max)) < threshold) {
            array[i] = 0;
        }
    }
}
