#ifndef MATRIX_CPU_H
#define MATRIX_CPU_H

void matrix_add_cpu(const float* A, const float* B, float* C, int size);
void matrix_multiply_cpu(const float* A, const float* B, float* C, int n);

#endif // MATRIX_CPU_H
