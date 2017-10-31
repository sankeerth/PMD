Info on arguments to be given in input file for POD and Sparse Coding
All source files are in 'src' directory
Makefile and input file argument for executable in 'bin' directory

Command to Run the executable is:
For single process => <path to executable> <path to input file>
For multiple processes => mpiexec -np <num_procs> <path to executable> <path to input file>
For using map command with multiple procs => map --profile -np <num_procs> <path to executable> <path to input file>

Format of 'pmd_input_file':
==== Controller ====
1. Task: '0' for POD and '1' for Sparse Coding
2. Job: If Task is '0':
            Job '0': PODSerial,
            Job '1': PODParallelColCyclic
            Job '2': PODParallelRowCyclic
            Job '3': PODReconstructCoefficientsFromModes
        If Task is '1':
            Job '0': SparseCodingSerialDenseMatrix
            Job '1': SparseCodingSerialSparseMatrix
            Job '2': SparseCodingParallelDenseMatrix

3. If '1', then compute POD with data in transposed fashion. This implementation is a mix of ColCyclic and RowCyclic.
It is recommended to use this option with PODParallelColCyclic for better usage of memory and speed of execution

==== Snapshot Data Info ====
4. Number of snapshots
5. Number of modes
6. Number of dimensions in the data (we are using 3D data, so 3 for now)
7. imax jmax kmax (Separated by a space, Should not use commas as separators or any other characters)
8. it_min it_max jt_min jt_max kt_min kt_max (Separated by a space, Should not use commas as separators or any other characters)

==== File Info ====
9. Max number of snapshot files that can be computed. This essentially forms the suffix of snapshot filename
10. File interval for skipping the snapshot files that are not required
11. Start index of the snapshots as to where to start the reading of snapshots from
12. Path of mesh file. The whole path which includes the name of the file as well
13. Path to snapshot files directory (append a '/' at the end as I do not in the code)
14. Prefix of the snapshot files, for ex: 'Sol3C'
15. Extension of the solution/snapshot files, for ex: '.q'. If extension not required, then just put '0' in that line
16. Path to output directory (Eigen values, POD bases and POD coefficients will be written here. Create this directory and append a '/' at the end as well)

==== Reconstuction Error Info ====
17. '1' to compute POD reconstruction error and '0' otherwise
18. '1' to compute Sparse Coding reconstruction error and '0' otherwise

==== Output Info ====
19. A boolean to represent if POD modes have to be written to in text format. '0' if not to be written. '1' if it has to be written.
20. A boolean to represent if POD modes have to be written to in binary format. '0' if not to be written. '1' if it has to be written.
21. A boolean to represent if POD modes and RMS error have to be written to in binary format. '0' if not to be written. '1' if it has to be written.

==== Config Info ====
22. sparsity used in Sparse Coding
23. Epsilon rank: This represents the value where all the snapshots between the largest and smallest value whose difference is less than the epsilon value are included in the rank.
