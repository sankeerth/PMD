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

SRCS = ../src/main.cpp \
	   ../src/common/sources/parser.cpp \
	   ../src/common/sources/scalapack_helpers.cpp \
	   ../src/common/sources/utilities.cpp \
	   ../src/controller/sources/task.cpp \
	   ../src/controller/sources/job.cpp \
	   ../src/POD/pod.cpp \
	   ../src/POD/common/base/pod_base.cpp \
	   ../src/POD/common/io/read/pod_io_read.cpp \
	   ../src/POD/common/io/write/pod_io_write.cpp \
	   ../src/POD/common/preprocessing/pod_preprocessing.cpp \
	   ../src/POD/common/postprocessing/error/pod_postprocessing_error.cpp \
	   ../src/POD/common/postprocessing/cleanup/pod_postprocessing_cleanup.cpp \
	   ../src/POD/common/utilities/pod_utilities.cpp \
	   ../src/POD/PODColCyclic/io/read/pod_col_cyclic_io_read.cpp \
	   ../src/POD/PODColCyclic/io/write/pod_col_cyclic_io_write.cpp \
	   ../src/POD/PODColCyclic/preprocessing/pod_col_cyclic_preprocessing.cpp \
	   ../src/POD/PODColCyclic/postprocessing/error/pod_col_cyclic_postprocessing_error.cpp \
	   ../src/POD/PODRowCyclic/io/read/pod_row_cyclic_io_read.cpp \
	   ../src/POD/PODRowCyclic/io/write/pod_row_cyclic_io_write.cpp \
	   ../src/POD/PODRowCyclic/preprocessing/pod_row_cyclic_preprocessing.cpp \
	   ../src/POD/PODRowCyclic/postprocessing/error/pod_row_cyclic_postprocessing_error.cpp \
	   ../src/SparseCoding/sparse_coding.cpp \
	   ../src/SparseCoding/common/io/read/sparse_coding_parallel_io_read.cpp \
	   ../src/SparseCoding/common/preprocessing/sparse_coding_preprocessing.cpp \
	   ../src/SparseCoding/common/postprocessing/error/sparse_coding_postprocessing_error.cpp \
	   ../src/SparseCoding/common/postprocessing/cleanup/sparse_coding_postprocessing_cleanup.cpp \
	   ../src/SparseCoding/common/utilites/sparse_coding_utilities.cpp \
	   ../src/SparseCoding/parallel/base/sparse_coding_parallel_base.cpp \

#SUBDIRS := $(shell ls ../src/ -F | grep "\/" )
#DIRS := $(SUBDIRS)
#SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cpp) )

OBJS = $(subst .cpp,.o,$(SRCS))

all: $(EXE)

%.o: ./%.cpp
	$(CXX) $(CCFLAGS) -I$(MKLINC) -c -o $@ $<

$(EXE): $(OBJS)
	$(CXX) -o $(EXE) $(OBJS) $(LIBS)

clean:
	rm $(OBJS) $(EXE)
