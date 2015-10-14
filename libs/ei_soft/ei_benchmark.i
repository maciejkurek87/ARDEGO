%module ei_soft
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1, double* IN_ARRAY2) {(int max_mius, double miu_mean[], double miu_s2[])};

%include "ei_bench.h"

