%module ei_soft
%{
#define SWIG_FILE_WITH_INIT
#include "ei_multi.h"
%}
%include "numpy.i"
%init %{
import_array();
%}

%apply ( int DIM1, double* IN_ARRAY1) {(int max_mius1, double *miu_mean), (int max_mius2, double *miu_s2), (int max_lambdas1, double *lambda_s2), (int max_lambdas2, double *lambda_mean)}

%include "ei_multi.h"
%rename (ei_multi_min) my_ei_multi_min;
%rename (ei_multi_max) my_ei_multi_max;

%inline %{
    double my_ei_multi_max(int max_mius1, double *miu_mean, int max_mius2, double *miu_s2, int max_lambdas1, double *lambda_s2, int max_lambdas2, double *lambda_mean, double y_best, int n){
    int nn = ((max_mius1 + max_lambdas2) * (max_mius1 + max_lambdas2));
    if (max_lambdas1 != ((max_mius1 + max_lambdas2) * (max_mius1 + max_lambdas2)) ) {
        PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given", max_lambdas2, nn);
        return 0.0;
    }
    if (max_mius2 != 0) {
        PyErr_Format(PyExc_ValueError, "Array max_mius2 should be empty");
        return 0.0;
    }
    return ei_multi_max(max_mius1, miu_mean, miu_s2, max_lambdas2, lambda_s2, lambda_mean, y_best, n);
}
%}

%inline %{
    double my_ei_multi_min(int max_mius1, double *miu_mean, int max_mius2, double *miu_s2, int max_lambdas1, double *lambda_s2, int max_lambdas2, double *lambda_mean, double y_best, int n){
    int nn = ((max_mius1 + max_lambdas2) * (max_mius1 + max_lambdas2));
    if (max_lambdas1 != ((max_mius1 + max_lambdas2) * (max_mius1 + max_lambdas2) )) {
        PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given", max_lambdas2, nn);
        return 0.0;
    }
    if (max_mius2 != 0) {
        PyErr_Format(PyExc_ValueError, "Array max_mius2 should be empty");
        return 0.0;
    }
    return ei_multi_min(max_mius1, miu_mean, miu_s2, max_lambdas2, lambda_s2, lambda_mean, y_best, n);
}
%}