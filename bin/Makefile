print-%  : ; @echo $* = $($*)

CXX = /usr/bin/g++
CXX = mpicc

# below variables are taken from pblasc in /opt/intel/mkl
ROOTdir = /opt/intel/mkl
MKLINC = $(ROOTdir)/include

# dynamic loading
LIBS = -L$(ROOTdir)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

CCFLAGS = -w -g -std=c++0x -u -O0 -DLOGS_RANK_ON

VERSION     = 3.11.1

ifeq ($(dbg), 1)
    TARGET = debug
else
    TARGET = release
endif

EXE = POD-$(VERSION)-$(TARGET)

vpath %.cpp ../src ../src/common/sources ../src/controller/sources ../src/POD ../src/POD/common/base ../src/POD/common/io/read ../src/POD/common/io/write ../src/POD/common/preprocessing ../src/POD/common/postprocessing/error ../src/POD/common/postprocessing/cleanup ../src/POD/common/utilities ../src/POD/PODColCyclic/io/read ../src/POD/PODColCyclic/io/write ../src/POD/PODColCyclic/preprocessing ../src/POD/PODColCyclic/postprocessing/error ../src/POD/PODRowCyclic/io/read ../src/POD/PODRowCyclic/io/write ../src/POD/PODRowCyclic/preprocessing ../src/POD/PODRowCyclic/postprocessing/error ../src/SparseCoding ../src/SparseCoding/common/io/read ../src/SparseCoding/common/preprocessing ../src/SparseCoding/common/postprocessing/error ../src/SparseCoding/common/postprocessing/cleanup ../src/SparseCoding/common/utilites ../src/SparseCoding/parallel/base 

src = main.cpp parser.cpp scalapack_helpers.cpp utilities.cpp task.cpp job.cpp pod.cpp pod_base.cpp pod_io_read.cpp pod_io_write.cpp pod_preprocessing.cpp pod_postprocessing_error.cpp pod_postprocessing_cleanup.cpp pod_utilities.cpp pod_col_cyclic_io_read.cpp pod_col_cyclic_io_write.cpp pod_col_cyclic_preprocessing.cpp pod_col_cyclic_postprocessing_error.cpp pod_row_cyclic_io_read.cpp pod_row_cyclic_io_write.cpp pod_row_cyclic_preprocessing.cpp pod_row_cyclic_postprocessing_error.cpp sparse_coding.cpp sparse_coding_parallel_io_read.cpp sparse_coding_preprocessing.cpp sparse_coding_postprocessing_error.cpp sparse_coding_postprocessing_cleanup.cpp sparse_coding_utilities.cpp sparse_coding_parallel_base.cpp

OBJS = $(patsubst %.cpp, %.o, $(src))

#SEARCH := $(shell ls ./ -F | grep "\/")
#DIRS := ../src/
#SUBDIRS := $(foreach d, $(DIRS), $(SEARCH))


all: $(EXE)

%.o: ./%.cpp
	$(CXX) $(CCFLAGS) -I$(MKLINC) -c -o $@ $<

$(EXE): $(OBJS)
	$(CXX) -o $(EXE) $(OBJS) $(LIBS)

clean:
	rm $(OBJS) $(EXE)
