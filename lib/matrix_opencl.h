#ifndef MATRIX_OPENCL_H
#define MATRIX_OPENCL_H

double matrix_add_opencl(const float* A, const float* B, float* C, int size);
double matrix_multiply_opencl(const float* A, const float* B, float* C, int n);

#endif // MATRIX_OPENCL_H
